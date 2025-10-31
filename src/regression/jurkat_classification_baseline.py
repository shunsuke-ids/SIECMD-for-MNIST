#!/usr/bin/env python3
"""
Jurkat (CellCycle) 7-class classification baseline using the same backbone as regression.

- Classes: G1, S, G2, Prophase, Metaphase, Anaphase, Telophase (7 classes, no merge)
- Data: Ch3 (brightfield) grayscale JPEGs resized to 66x66 by default
- Model: Same CNN backbone; softmax head (7-way), loss: sparse_categorical_crossentropy
- Metrics: classification accuracy（ラベルでの純粋な分類）。

角度は扱いません。`mnist_fine_tuning.py`（回帰）と `mnist_classification.py`（分類）の関係と同様に、
本ファイルは「ラベル分類のみ」を担います。
"""
import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict

from keras import models as km, layers as kl
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

try:
    from sklearn.metrics import classification_report, confusion_matrix  # optional
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

PHASE_DIR = Path('/home/shunsuke/data/raw/extracted/CellCycle')
PHASES7 = ['G1','S','G2','Prophase','Metaphase','Anaphase','Telophase']
ANGLE_STEP_7 = 360.0 / len(PHASES7)


def load_ch3_manifest(limit_per_phase: int = None, image_size: int = 66) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    phase_labels: List[str] = []
    phase_counts: Dict[str, int] = {p: 0 for p in PHASES7}

    patterns = ['*Ch3*.jpg', '*Ch3*.jpeg']
    for ph in PHASES7:
        files: List[Path] = []
        for pat in patterns:
            files.extend(sorted((PHASE_DIR / ph).glob(pat)))
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
    return X, np.array(phase_labels)


def train_val_test_split(X, labels, val_ratio=0.15, test_ratio=0.15, seed=42):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    test_size = int(N*test_ratio)
    val_size = int(N*val_ratio)
    test_idx = idx[:test_size]
    val_idx = idx[test_size:test_size+val_size]
    train_idx = idx[test_size+val_size:]
    return (X[train_idx], X[val_idx], X[test_idx]), (labels[train_idx], labels[val_idx], labels[test_idx])


def create_backbone(input_shape=(128,128,1)):
    inputs = kl.Input(shape=input_shape)
    x = kl.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = kl.GlobalAveragePooling2D()(x)
    x = kl.Dense(256, activation='relu')(x)
    features = kl.Dense(128, activation='relu')(x)
    return inputs, features


def create_classification_model(input_shape=(128,128,1), num_classes=7):
    inputs, features = create_backbone(input_shape)
    outputs = kl.Dense(num_classes, activation='softmax')(features)
    return km.Model(inputs, outputs)


def main():
    parser = argparse.ArgumentParser(description='Jurkat 7-class classification baseline (same backbone)')
    parser.add_argument('--epochs','-e',type=int,default=20)
    parser.add_argument('--batch_size','-b',type=int,default=64)
    parser.add_argument('--runs','-r',type=int,default=1)
    parser.add_argument('--folds',type=int,default=1,help='If >1, perform stratified K-fold CV with this number of folds (runs is ignored).')
    parser.add_argument('--limit_per_phase',type=int,default=None,help='Optional cap per phase for rapid prototype.')
    parser.add_argument('--image_size',type=int,default=66)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--save_weights', action='store_true', help='Save best val_accuracy weights and load for test eval.')
    parser.add_argument('--confmat', action='store_true', help='Compute and save confusion matrix for the test set of each fold/run.')
    parser.add_argument('--confmat_norm', choices=['none','true','pred','all'], default='none', help='Normalization for confusion matrix (sklearn).')
    parser.add_argument('--out_dir', type=str, default='results/confusion_matrices/classification', help='Directory to save confusion matrices.')
    args = parser.parse_args()

    print('=== Jurkat 7-class classification baseline ===')
    print(f'Phases: {PHASES7}')
    print(f'Image size: {args.image_size}x{args.image_size}  | Channel: Ch3 (brightfield)')
    print(f'Limit per phase: {args.limit_per_phase}')

    all_cls_acc=[]
    X, labels = load_ch3_manifest(limit_per_phase=args.limit_per_phase, image_size=args.image_size) # X：画像データ、labels：対応するラベル
    label_to_idx = {p:i for i,p in enumerate(PHASES7)} # ラベルを整数に変換する辞書
    y_all_idx = np.array([label_to_idx[l] for l in labels], dtype=np.int32) # 整数のみの配列に変換

    if args.folds and args.folds > 1:
        print(f"Using stratified {args.folds}-fold cross-validation. 'runs' is ignored.")
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_num = 0
        for train_idx, test_idx in skf.split(X, y_all_idx):
            fold_num += 1
            print(f'Fold {fold_num}/{args.folds}')
            X_train_full, X_test = X[train_idx], X[test_idx]
            y_train_full, y_test = y_all_idx[train_idx], y_all_idx[test_idx]

            # Create a stratified validation split from training fold (15%)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed+fold_num)
            tr_idx, val_idx = next(sss.split(X_train_full, y_train_full))
            Xtr, Xval = X_train_full[tr_idx], X_train_full[val_idx]
            ytr_idx, yval_idx = y_train_full[tr_idx], y_train_full[val_idx]

            print(f'Data shapes: train={Xtr.shape}, val={Xval.shape}, test={X_test.shape}')

            model = create_classification_model(input_shape=(args.image_size,args.image_size,1), num_classes=len(PHASES7))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            callbacks = []
            weights_path = None
            if args.save_weights:
                ckpt_dir = Path('weights/jurkat_7cls_cls'); ckpt_dir.mkdir(parents=True, exist_ok=True)
                weights_path = ckpt_dir / f'fold{fold_num}.keras'
                mc = ModelCheckpoint(str(weights_path), save_best_only=True, monitor='val_accuracy', mode='max', verbose=1, save_weights_only=True)
                callbacks.append(mc)

            _ = model.fit(Xtr, ytr_idx, validation_data=(Xval, yval_idx), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1)

            if args.save_weights and weights_path and weights_path.exists():
                print('Loading best weights for evaluation...')
                model.load_weights(str(weights_path))

            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
            all_cls_acc.append(float(test_acc))
            print(f'Fold {fold_num} Test Accuracy: {test_acc*100:.2f}%')

            if fold_num == args.folds and _HAS_SKLEARN:
                probs = model.predict(X_test, verbose=0)
                y_pred_idx = np.argmax(probs, axis=1)
                print('Classification report (last fold):\n')
                print(classification_report(y_test, y_pred_idx, target_names=PHASES7, digits=4))

            # Confusion matrix per fold
            if args.confmat and _HAS_SKLEARN:
                probs = model.predict(X_test, verbose=0)
                y_pred_idx = np.argmax(probs, axis=1)
                norm = None if args.confmat_norm == 'none' else args.confmat_norm
                cm = confusion_matrix(y_test, y_pred_idx, labels=list(range(len(PHASES7))), normalize=norm)
                # Save
                out_dir = Path(args.out_dir) / f'fold{fold_num}'
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(out_dir / 'confusion_matrix.csv', cm, delimiter=',', fmt='%0.6f')
                print(f'Confusion matrix saved to {out_dir / "confusion_matrix.csv"}')
    else:
        for run in range(args.runs):
            print(f'Run {run+1}/{args.runs}')
            (Xtr,Xval,Xte),(ltr,lval,lte) = train_val_test_split(X, labels, seed=args.seed+run)
            print(f'Data shapes: train={Xtr.shape}, val={Xval.shape}, test={Xte.shape}')

            ytr_idx = np.array([label_to_idx[l] for l in ltr], dtype=np.int32)
            yval_idx = np.array([label_to_idx[l] for l in lval], dtype=np.int32)
            yte_idx = np.array([label_to_idx[l] for l in lte], dtype=np.int32)

            model = create_classification_model(input_shape=(args.image_size,args.image_size,1), num_classes=len(PHASES7))
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            callbacks = []
            weights_path = None
            if args.save_weights:
                ckpt_dir = Path('weights/jurkat_7cls_cls'); ckpt_dir.mkdir(parents=True, exist_ok=True)
                weights_path = ckpt_dir / f'run{run}.keras'
                mc = ModelCheckpoint(str(weights_path), save_best_only=True, monitor='val_accuracy', mode='max', verbose=1, save_weights_only=True)
                callbacks.append(mc)

            _ = model.fit(Xtr, ytr_idx, validation_data=(Xval, yval_idx), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1)

            if args.save_weights and weights_path and weights_path.exists():
                print('Loading best weights for evaluation...')
                model.load_weights(str(weights_path))

            test_loss, test_acc = model.evaluate(Xte, yte_idx, verbose=0)
            all_cls_acc.append(float(test_acc))
            print(f'Run {run+1} Test Accuracy: {test_acc*100:.2f}%')

            if run == args.runs - 1 and _HAS_SKLEARN:
                probs = model.predict(Xte, verbose=0)
                y_pred_idx = np.argmax(probs, axis=1)
                print('Classification report:\n')
                print(classification_report(yte_idx, y_pred_idx, target_names=PHASES7, digits=4))

            # Confusion matrix per run (single-split mode)
            if args.confmat and _HAS_SKLEARN:
                probs = model.predict(Xte, verbose=0)
                y_pred_idx = np.argmax(probs, axis=1)
                norm = None if args.confmat_norm == 'none' else args.confmat_norm
                cm = confusion_matrix(yte_idx, y_pred_idx, labels=list(range(len(PHASES7))), normalize=norm)
                out_dir = Path(args.out_dir) / f'run{run+1}'
                out_dir.mkdir(parents=True, exist_ok=True)
                np.savetxt(out_dir / 'confusion_matrix.csv', cm, delimiter=',', fmt='%0.6f')
                print(f'Confusion matrix saved to {out_dir / "confusion_matrix.csv"}')

    print('--- Summary ---')
    n = args.folds if (args.folds and args.folds > 1) else args.runs
    print(f'Accuracies: {[f"{a*100:.2f}%" for a in all_cls_acc]}')
    print(f'Mean Accuracy: {np.mean(all_cls_acc)*100:.2f}% ± {np.std(all_cls_acc)*100:.2f}% (n={n})')

if __name__ == '__main__':
    main()
