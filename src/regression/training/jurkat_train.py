#!/usr/bin/env python3
"""
Unified Jurkat (CellCycle) training script for both classification and circular regression.

Usage:
    # Classification (7-class softmax)
    python jurkat_train.py --task classification --epochs 20 --folds 5

    # Circular Regression (7-class on unit circle)
    python jurkat_train.py --task regression --epochs 20 --folds 5

Dataset: Ch3 (brightfield) grayscale images, 7 phases
Classes: G1, S, G2, Prophase, Metaphase, Anaphase, Telophase
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from sklearn.metrics import classification_report
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.DL.metrics import prediction_mean_deviation
from src.DL.losses import linear_dist_squared_loss
from src.regression.utils.format_gt import (
    points_2_angles, associated_points_on_circle, angles_2_unit_circle_points,
    build_angle_mapping_equal, angle_to_label_with_mapping
)
from src.regression.utils.data_loaders import load_jurkat_ch3_data, get_label_to_index_mapping, PHASES7
from src.regression.utils.model_builders import create_jurkat_classification_model, create_jurkat_regression_model
from src.regression.utils.training_utils import setup_checkpoint, save_confusion_matrix, tolerance_accuracy


def build_angle_mapping():
    return build_angle_mapping_equal(PHASES7, start_angle=0)


def train_val_test_split_simple(X, labels, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Simple random split for non-CV mode"""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)
    rng.shuffle(idx)
    test_size = int(N * test_ratio)
    val_size = int(N * val_ratio)
    test_idx = idx[:test_size]
    val_idx = idx[test_size:test_size + val_size]
    train_idx = idx[test_size + val_size:]
    return (X[train_idx], X[val_idx], X[test_idx]), (labels[train_idx], labels[val_idx], labels[test_idx])


def main():
    parser = argparse.ArgumentParser(description='Jurkat unified training (classification/regression)')
    parser.add_argument('--task', choices=['classification', 'regression'], required=True,
                       help='Task type: classification (softmax) or regression (circular)')
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--runs', '-r', type=int, default=1)
    parser.add_argument('--folds', type=int, default=1,
                       help='If >1, perform stratified K-fold CV (runs is ignored)')
    parser.add_argument('--limit_per_phase', type=int, default=None)
    parser.add_argument('--image_size', type=int, default=66)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--confmat', action='store_true', help='Save confusion matrix')
    parser.add_argument('--confmat_norm', choices=['none', 'true', 'pred', 'all'], default='none')
    parser.add_argument('--out_dir', type=str, default='results/confusion_matrices')
    # Regression-specific
    parser.add_argument('--tolerance', type=float, default=None,
                       help='Tolerance for regression (default: 180/7 degrees)')
    args = parser.parse_args()

    is_regression = (args.task == 'regression')
    angle_mapping = build_angle_mapping() if is_regression else None

    if is_regression and args.tolerance is None:
        args.tolerance = 180.0 / 7.0

    print(f'=== Jurkat 7-class {args.task} ===')
    print(f'Phases: {PHASES7}')
    print(f'Image size: {args.image_size}x{args.image_size}')
    if is_regression:
        print(f'Angle mapping: {angle_mapping}')
        print(f'Tolerance: ±{args.tolerance:.2f}°')

    # Load data
    X, labels = load_jurkat_ch3_data(limit_per_phase=args.limit_per_phase, image_size=args.image_size)
    label_to_idx = get_label_to_index_mapping(PHASES7)
    y_all_idx = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

    if is_regression:
        angles_all = np.array([angle_mapping[l] for l in labels], dtype=np.float32)

    # Results storage
    all_results = []

    # Cross-validation mode
    if args.folds and args.folds > 1:
        if not _HAS_SKLEARN:
            raise RuntimeError('scikit-learn required for K-fold CV')
        print(f'Using stratified {args.folds}-fold CV')
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        fold_num = 0

        for train_idx, test_idx in skf.split(X, y_all_idx):
            fold_num += 1
            print(f'\nFold {fold_num}/{args.folds}')

            X_train_full, X_test = X[train_idx], X[test_idx]
            yidx_train_full, yidx_test = y_all_idx[train_idx], y_all_idx[test_idx]

            # Stratified val split
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed + fold_num)
            tr_idx, val_idx = next(sss.split(X_train_full, yidx_train_full))
            Xtr, Xval = X_train_full[tr_idx], X_train_full[val_idx]
            ytr_idx, yval_idx = yidx_train_full[tr_idx], yidx_train_full[val_idx]

            print(f'Data: train={Xtr.shape}, val={Xval.shape}, test={X_test.shape}')

            # Prepare targets based on task
            if is_regression:
                ang_train_full = angles_all[train_idx]
                atr, aval = ang_train_full[tr_idx], ang_train_full[val_idx]
                ang_test = angles_all[test_idx]
                ytr = angles_2_unit_circle_points(atr)
                yval = angles_2_unit_circle_points(aval)
                model = create_jurkat_regression_model(input_shape=(args.image_size, args.image_size, 1))
                model.compile(optimizer='adam', loss=linear_dist_squared_loss)
                monitor, mode = 'val_loss', 'min'
            else:
                ytr, yval = ytr_idx, yval_idx
                model = create_jurkat_classification_model(input_shape=(args.image_size, args.image_size, 1),
                                                          num_classes=len(PHASES7))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                monitor, mode = 'val_accuracy', 'max'

            # Checkpoint
            callbacks = []
            if args.save_weights:
                subdir = 'jurkat_7cls' if is_regression else 'jurkat_7cls_cls'
                mc = setup_checkpoint(f'weights/{subdir}', f'fold{fold_num}.keras',
                                     monitor=monitor, mode=mode, save_weights_only=(not is_regression))
                callbacks.append(mc)

            # Train
            model.fit(Xtr, ytr, validation_data=(Xval, yval),
                     epochs=args.epochs, batch_size=args.batch_size,
                     callbacks=callbacks, verbose=1)

            # Evaluate
            if is_regression:
                preds = model.predict(X_test, verbose=0)
                preds = associated_points_on_circle(preds)
                pred_angles = points_2_angles(preds)
                mean_dev = prediction_mean_deviation(ang_test, pred_angles)
                tol_acc = tolerance_accuracy(pred_angles, ang_test, tol=args.tolerance)
                all_results.append(mean_dev)
                print(f'Fold {fold_num}: deviation={mean_dev:.2f}°, tol_acc@±{args.tolerance:.2f}°={tol_acc*100:.2f}%')

                if args.confmat and _HAS_SKLEARN:
                    pred_labels = [angle_to_label_with_mapping(a, angle_mapping) for a in pred_angles]
                    y_pred_idx = np.array([label_to_idx[l] for l in pred_labels], dtype=np.int32)
                    save_confusion_matrix(yidx_test, y_pred_idx, PHASES7,
                                        f'{args.out_dir}/regression', args.confmat_norm, f'fold{fold_num}')
            else:
                test_loss, test_acc = model.evaluate(X_test, yidx_test, verbose=0)
                all_results.append(test_acc)
                print(f'Fold {fold_num}: accuracy={test_acc*100:.2f}%')

                if fold_num == args.folds and _HAS_SKLEARN:
                    probs = model.predict(X_test, verbose=0)
                    y_pred_idx = np.argmax(probs, axis=1)
                    print('\nClassification report (last fold):')
                    print(classification_report(yidx_test, y_pred_idx, target_names=PHASES7, digits=4))

                if args.confmat and _HAS_SKLEARN:
                    probs = model.predict(X_test, verbose=0)
                    y_pred_idx = np.argmax(probs, axis=1)
                    save_confusion_matrix(yidx_test, y_pred_idx, PHASES7,
                                        f'{args.out_dir}/classification', args.confmat_norm, f'fold{fold_num}')

    # Simple runs mode
    else:
        for run in range(args.runs):
            print(f'\nRun {run+1}/{args.runs}')

            if is_regression:
                (Xtr, Xval, Xte), (ltr, lval, lte) = train_val_test_split_simple(X, labels, seed=args.seed + run)
                atr = np.array([angle_mapping[l] for l in ltr], dtype=np.float32)
                aval = np.array([angle_mapping[l] for l in lval], dtype=np.float32)
                ate = np.array([angle_mapping[l] for l in lte], dtype=np.float32)
                ytr = angles_2_unit_circle_points(atr)
                yval = angles_2_unit_circle_points(aval)
                yte_idx = np.array([label_to_idx[l] for l in lte], dtype=np.int32)
                model = create_jurkat_regression_model(input_shape=(args.image_size, args.image_size, 1))
                model.compile(optimizer='adam', loss=linear_dist_squared_loss)
                monitor, mode = 'val_loss', 'min'
            else:
                (Xtr, Xval, Xte), (ltr, lval, lte) = train_val_test_split_simple(X, labels, seed=args.seed + run)
                ytr = np.array([label_to_idx[l] for l in ltr], dtype=np.int32)
                yval = np.array([label_to_idx[l] for l in lval], dtype=np.int32)
                yte_idx = np.array([label_to_idx[l] for l in lte], dtype=np.int32)
                model = create_jurkat_classification_model(input_shape=(args.image_size, args.image_size, 1),
                                                          num_classes=len(PHASES7))
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                monitor, mode = 'val_accuracy', 'max'

            print(f'Data: train={Xtr.shape}, val={Xval.shape}, test={Xte.shape}')

            # Checkpoint
            callbacks = []
            if args.save_weights:
                subdir = 'jurkat_7cls' if is_regression else 'jurkat_7cls_cls'
                mc = setup_checkpoint(f'weights/{subdir}', f'run{run}.keras',
                                     monitor=monitor, mode=mode, save_weights_only=(not is_regression))
                callbacks.append(mc)

            # Train
            model.fit(Xtr, ytr, validation_data=(Xval, yval),
                     epochs=args.epochs, batch_size=args.batch_size,
                     callbacks=callbacks, verbose=1)

            # Evaluate
            if is_regression:
                preds = model.predict(Xte, verbose=0)
                preds = associated_points_on_circle(preds)
                pred_angles = points_2_angles(preds)
                mean_dev = prediction_mean_deviation(ate, pred_angles)
                tol_acc = tolerance_accuracy(pred_angles, ate, tol=args.tolerance)
                all_results.append(mean_dev)
                print(f'Run {run+1}: deviation={mean_dev:.2f}°, tol_acc={tol_acc*100:.2f}%')

                if args.confmat and _HAS_SKLEARN:
                    pred_labels = [angle_to_label_with_mapping(a, angle_mapping) for a in pred_angles]
                    y_pred_idx = np.array([label_to_idx[l] for l in pred_labels], dtype=np.int32)
                    save_confusion_matrix(yte_idx, y_pred_idx, PHASES7,
                                        f'{args.out_dir}/regression', args.confmat_norm, f'run{run+1}')
            else:
                test_loss, test_acc = model.evaluate(Xte, yte_idx, verbose=0)
                all_results.append(test_acc)
                print(f'Run {run+1}: accuracy={test_acc*100:.2f}%')

                if run == args.runs - 1 and _HAS_SKLEARN:
                    probs = model.predict(Xte, verbose=0)
                    y_pred_idx = np.argmax(probs, axis=1)
                    print('\nClassification report:')
                    print(classification_report(yte_idx, y_pred_idx, target_names=PHASES7, digits=4))

                if args.confmat and _HAS_SKLEARN:
                    probs = model.predict(Xte, verbose=0)
                    y_pred_idx = np.argmax(probs, axis=1)
                    save_confusion_matrix(yte_idx, y_pred_idx, PHASES7,
                                        f'{args.out_dir}/classification', args.confmat_norm, f'run{run+1}')

    # Summary
    print('\n=== Summary ===')
    n = args.folds if (args.folds and args.folds > 1) else args.runs
    if is_regression:
        print(f'Mean deviation: {np.mean(all_results):.2f} ± {np.std(all_results):.2f}° (n={n})')
    else:
        print(f'Accuracies: {[f"{a*100:.2f}%" for a in all_results]}')
        print(f'Mean accuracy: {np.mean(all_results)*100:.2f} ± {np.std(all_results)*100:.2f}% (n={n})')


if __name__ == '__main__':
    main()
