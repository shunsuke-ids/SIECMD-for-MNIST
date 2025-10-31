#!/usr/bin/env python3
"""
Jurkat (CellCycle) 7-class circular regression

Classes:
    G1, S, G2, Prophase, Metaphase, Anaphase, Telophase = 7 (no merging)

Approach:
    Map 7 classes onto unit circle with equal spacing of 360/7 degrees (non-integer centers are OK).
    Train a CNN to regress a 2D point on (approx) unit circle using linear_dist_squared_loss.
    Evaluate mean angular deviation and tolerance accuracy.
    Load brightfield channel (Ch3) images only.

Input images:
    - Brightfield (Ch3): JPEG files whose names include "Ch3" (e.g., *Ch3.ome.jpg, *Ch3.jpg)
    - Images are grayscale. Default resize to 66x66 (original size); can be changed via --image_size.

Notes:
    - Non-integer angle centers are fine (cos/sin accept floats).
    - Default tolerance is half of inter-class spacing (180/7 ≈ 25.714°).
    - Extremely low samples for Metaphase/Anaphase/Telophase; this script does NOT yet apply focal loss or oversampling.
"""
import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict
try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

from keras import models as km, layers as kl, callbacks
from keras.callbacks import ModelCheckpoint
try:
    from sklearn.metrics import confusion_matrix
    _HAS_SKLEARN_METRICS = True
except Exception:
    _HAS_SKLEARN_METRICS = False

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.DL.metrics import prediction_mean_deviation
from src.DL.activation_functions import sigmoid_activation
from src.DL.losses import linear_dist_squared_loss
from src.regression.format_gt import points_2_angles, associated_points_on_circle, angles_2_unit_circle_points

PHASE_DIR = Path('/home/shunsuke/data/raw/extracted/CellCycle')
PHASES7 = ['G1','S','G2','Prophase','Metaphase','Anaphase','Telophase']
ANGLE_STEP_7 = 360.0 / len(PHASES7)

def build_angle_mapping():
    return {phase: (i * ANGLE_STEP_7) % 360.0 for i, phase in enumerate(PHASES7)}

def build_angle_arrays(labels: List[str], angle_mapping: Dict[str, float]) -> np.ndarray:
    angles = np.array([angle_mapping[l] for l in labels], dtype=np.float32)
    return angles

def load_ch3_manifest(limit_per_phase: int = None, image_size: int = 66) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load brightfield (Ch3) images for 7 phases.

    Args:
        limit_per_phase: optional cap per phase for quick prototyping.
        image_size: side length for resize (square). Default 66 (original dataset size).
    Returns:
        X (N,H,W,1), labels_for_angles (N,), labels (N,)
    """
    X: List[np.ndarray] = []
    phase_labels: List[str] = []
    phase_counts: Dict[str, int] = {p: 0 for p in PHASES7}

    # Brightfield Ch3 JPEG files (filenames may include ".ome" but are .jpg/jpeg)
    patterns = ['*Ch3*.jpg', '*Ch3*.jpeg']

    for ph in PHASES7:
        files: List[Path] = []
        for pat in patterns:
            files.extend(sorted((PHASE_DIR / ph).glob(pat)))
        # Deduplicate while preserving order
        seen = set()
        uniq_files = []
        for p in files:
            if p not in seen:
                uniq_files.append(p); seen.add(p)

        for p in uniq_files:
            if limit_per_phase is not None and phase_counts[ph] >= limit_per_phase:
                break
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[0] != image_size or img.shape[1] != image_size:
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            X.append(img[..., None])
            phase_labels.append(ph)
            phase_counts[ph] += 1

    X = np.stack(X, axis=0)
    return X, np.array(phase_labels), np.array(phase_labels)

def train_val_test_split(X, angles, labels, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    test_size = int(N*test_ratio)
    val_size = int(N*val_ratio)
    test_idx = idx[:test_size]
    val_idx = idx[test_size:test_size+val_size]
    train_idx = idx[test_size+val_size:]
    return (X[train_idx], X[val_idx], X[test_idx]), (angles[train_idx], angles[val_idx], angles[test_idx]), (labels[train_idx], labels[val_idx], labels[test_idx])

def create_cnn(input_shape=(128,128,1)):
    inputs = kl.Input(shape=input_shape)
    x = kl.Conv2D(32,3,padding='same',activation='relu')(inputs)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(64,3,padding='same',activation='relu')(x)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(128,3,padding='same',activation='relu')(x)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(256,3,padding='same',activation='relu')(x)
    x = kl.GlobalAveragePooling2D()(x)
    x = kl.Dense(256,activation='relu')(x)
    x = kl.Dense(128,activation='relu')(x)
    outputs = kl.Dense(2,activation=sigmoid_activation)(x)
    return km.Model(inputs, outputs)

def angle_to_label(angle: float, angle_mapping: Dict[str, float]) -> str:
    # choose closest mapped angle circularly
    best = None; best_dist=999.0
    for lbl,ang in angle_mapping.items():
        diff = abs(angle-ang)
        diff = min(diff, 360.0-diff)
        if diff < best_dist:
            best_dist=diff; best=lbl
    return best

def tolerance_accuracy(pred_angles: np.ndarray, true_angles: np.ndarray, tol: float) -> float:
    diffs = np.abs(pred_angles-true_angles)
    diffs = np.minimum(diffs, 360-diffs)
    return float(np.mean(diffs <= tol))

# snapped accuracy omitted by request; tolerance accuracy is used as the primary discrete-like metric.

def main():
    parser = argparse.ArgumentParser(description='Jurkat 7-class circular regression (no merge)')
    parser.add_argument('--epochs','-e',type=int,default=20)
    parser.add_argument('--batch_size','-b',type=int,default=64)
    parser.add_argument('--runs','-r',type=int,default=1)
    parser.add_argument('--folds',type=int,default=1, help='If >1, perform stratified K-fold CV on labels (runs is ignored).')
    parser.add_argument('--limit_per_phase',type=int,default=None,help='Optional cap per phase for rapid prototype.')
    parser.add_argument('--image_size',type=int,default=66)
    parser.add_argument('--tolerance',type=float,default=None, help='If None, set to half inter-class spacing (180/7).')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--confmat', action='store_true', help='Compute and save confusion matrix for the test set of each fold/run (angles snapped to nearest center).')
    parser.add_argument('--confmat_norm', choices=['none','true','pred','all'], default='none', help='Normalization for confusion matrix (sklearn).')
    parser.add_argument('--out_dir', type=str, default='results/confusion_matrices/regression', help='Directory to save confusion matrices.')
    args = parser.parse_args()

    angle_mapping = build_angle_mapping()
    if args.tolerance is None:
        args.tolerance = 180.0/7.0

    print('=== Jurkat 7-class circular regression ===')
    print(f'Phases: {PHASES7}')
    print(f'Angle step: {ANGLE_STEP_7:.6f}°')
    print(f'Angles mapping: {angle_mapping}')
    print(f'Image size: {args.image_size}x{args.image_size}  | Channel: Ch3 (brightfield)')
    print(f'Limit per phase: {args.limit_per_phase}')

    all_mean_dev=[]; all_tol_acc=[]

    # Load dataset once
    X, raw_labels_for_angles, labels = load_ch3_manifest(limit_per_phase=args.limit_per_phase, image_size=args.image_size)
    angles_all = build_angle_arrays(raw_labels_for_angles.tolist(), angle_mapping)
    label_to_idx = {p:i for i,p in enumerate(PHASES7)}
    y_all_idx = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

    if args.folds and args.folds > 1:
        if not _HAS_SKLEARN:
            raise RuntimeError('scikit-learn is required for stratified K-fold. Please install scikit-learn or run with --folds 1.')
        print(f'Using stratified {args.folds}-fold cross-validation. "runs" is ignored.')
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_num = 0
        for train_idx, test_idx in skf.split(X, y_all_idx):
            fold_num += 1
            print(f'Fold {fold_num}/{args.folds}')
            X_train_full, X_test = X[train_idx], X[test_idx]
            yidx_train_full, yidx_test = y_all_idx[train_idx], y_all_idx[test_idx]
            ang_train_full, ang_test = angles_all[train_idx], angles_all[test_idx]

            # Stratified val split (15% of training fold)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed+fold_num)
            tr_idx, val_idx = next(sss.split(X_train_full, yidx_train_full))
            Xtr, Xval = X_train_full[tr_idx], X_train_full[val_idx]
            atr, aval = ang_train_full[tr_idx], ang_train_full[val_idx]

            print(f'Data shapes: train={Xtr.shape}, val={Xval.shape}, test={X_test.shape}')
            ytr = angles_2_unit_circle_points(atr)
            yval = angles_2_unit_circle_points(aval)

            model = create_cnn(input_shape=(args.image_size,args.image_size,1))
            model.compile(optimizer='adam', loss=linear_dist_squared_loss)
            ckpt_dir = Path('weights/jurkat_7cls'); ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f'fold{fold_num}.keras'
            mc = ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor='val_loss', mode='min', verbose=1)
            _ = model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=args.epochs, batch_size=args.batch_size, callbacks=[mc], verbose=1)

            preds = model.predict(X_test, verbose=0)
            preds = associated_points_on_circle(preds)
            pred_angles = points_2_angles(preds)

            mean_dev = prediction_mean_deviation(ang_test, pred_angles)
            tol_acc = tolerance_accuracy(pred_angles, ang_test, tol=args.tolerance)
            all_mean_dev.append(mean_dev); all_tol_acc.append(tol_acc)
            print(f'Fold {fold_num} mean deviation: {mean_dev:.2f}°  tolerance@±{args.tolerance:.2f}°: {tol_acc*100:.2f}%')

            # Confusion matrix (snap predicted angles to nearest class center)
            if args.confmat and _HAS_SKLEARN_METRICS:
                pred_labels = [angle_to_label(a, angle_mapping) for a in pred_angles]
                y_pred_idx = np.array([label_to_idx[l] for l in pred_labels], dtype=np.int32)
                norm = None if args.confmat_norm == 'none' else args.confmat_norm
                cm = confusion_matrix(yidx_test, y_pred_idx, labels=list(range(len(PHASES7))), normalize=norm)
                out_dir = Path(args.out_dir) / f'fold{fold_num}'
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(out_dir / 'confusion_matrix.csv', cm, delimiter=',', fmt='%0.6f')
                print(f'Confusion matrix saved to {out_dir / "confusion_matrix.csv"}')

        print('--- Summary ---')
        print(f'Mean deviation: {np.mean(all_mean_dev):.2f} ± {np.std(all_mean_dev):.2f}°')
        print(f'Tolerance accuracy (±{args.tolerance}°): {np.mean(all_tol_acc)*100:.2f} ± {np.std(all_tol_acc)*100:.2f}%')
    else:
        for run in range(args.runs):
            print(f'Run {run+1}/{args.runs}')
            (Xtr,Xval,Xte),(atr,aval,ate),(ltr,lval,lte) = train_val_test_split(X, angles_all, labels, seed=args.seed+run)
            print(f'Data shapes: train={Xtr.shape}, val={Xval.shape}, test={Xte.shape}')
            ytr = angles_2_unit_circle_points(atr)
            yval = angles_2_unit_circle_points(aval)

            model = create_cnn(input_shape=(args.image_size,args.image_size,1))
            model.compile(optimizer='adam', loss=linear_dist_squared_loss)
            ckpt_dir = Path('weights/jurkat_7cls'); ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f'run{run}.keras'
            mc = ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor='val_loss', mode='min', verbose=1)
            _ = model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=args.epochs, batch_size=args.batch_size, callbacks=[mc], verbose=1)

            preds = model.predict(Xte, verbose=0)
            preds = associated_points_on_circle(preds)
            pred_angles = points_2_angles(preds)

            mean_dev = prediction_mean_deviation(ate, pred_angles)
            tol_acc = tolerance_accuracy(pred_angles, ate, tol=args.tolerance)
            all_mean_dev.append(mean_dev); all_tol_acc.append(tol_acc)
            print(f'Run {run+1} mean deviation: {mean_dev:.2f}°  tolerance@±{args.tolerance:.2f}°: {tol_acc*100:.2f}%')

            if args.confmat and _HAS_SKLEARN_METRICS:
                pred_labels = [angle_to_label(a, angle_mapping) for a in pred_angles]
                y_pred_idx = np.array([label_to_idx[l] for l in pred_labels], dtype=np.int32)
                norm = None if args.confmat_norm == 'none' else args.confmat_norm
                # Need true indices for test split
                label_to_idx_local = {p:i for i,p in enumerate(PHASES7)}
                yte_idx = np.array([label_to_idx_local[l] for l in lte], dtype=np.int32)
                cm = confusion_matrix(yte_idx, y_pred_idx, labels=list(range(len(PHASES7))), normalize=norm)
                out_dir = Path(args.out_dir) / f'run{run+1}'
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(out_dir / 'confusion_matrix.csv', cm, delimiter=',', fmt='%0.6f')
                print(f'Confusion matrix saved to {out_dir / "confusion_matrix.csv"}')

        print('--- Summary ---')
        print(f'Mean deviation: {np.mean(all_mean_dev):.2f} ± {np.std(all_mean_dev):.2f}°')
        print(f'Tolerance accuracy (±{args.tolerance}°): {np.mean(all_tol_acc)*100:.2f} ± {np.std(all_tol_acc)*100:.2f}%')

if __name__ == '__main__':
    main()