#!/usr/bin/env python3
"""
Unified Jurkat (CellCycle) training script for both classification and circular regression.

Usage:
    # Classification (7-class softmax)
    python jurkat_train.py --task classification --epochs 20 --folds 5

    # Circular Regression (7-class on unit circle)
    python jurkat_train.py --task regression --epochs 20 --folds 5

Dataset: Ch3 (brightfield) 66x66 grayscale images, 7 phases
Classes: G1, S, G2, Prophase, Metaphase, Anaphase, Telophase

Note: To change angle spacing, modify ANGLE_START in build_angle_mapping()
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from src.DL.metrics import prediction_mean_deviation
from src.DL.losses import linear_dist_squared_loss, cos_similarity_loss
from src.regression.utils.format_gt import (
    points_2_angles, associated_points_on_circle, angles_2_unit_circle_points,
    build_angle_mapping_equal, angle_to_label_with_mapping
)
from src.regression.utils.data_loaders import load_jurkat_ch3_data, get_label_to_index_mapping, PHASES7
from src.regression.utils.model_builders import create_jurkat_classification_model, create_jurkat_regression_model, create_jurkat_multitask_model
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


def prepare_split_data(use_cv, iteration, X, labels, y_all_idx, angles_all, is_regression,
                       label_to_idx, angle_mapping, seed, train_idx=None, test_idx=None):
    """Prepare train/val/test splits for either CV or simple run mode"""
    if use_cv:
        X_train_full, X_test = X[train_idx], X[test_idx]
        yidx_train_full, yidx_test = y_all_idx[train_idx], y_all_idx[test_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + iteration)
        tr_idx, val_idx = next(sss.split(X_train_full, yidx_train_full))
        Xtr, Xval = X_train_full[tr_idx], X_train_full[val_idx]

        if is_regression:
            ang_train_full = angles_all[train_idx]
            labels_tr, labels_val = ang_train_full[tr_idx], ang_train_full[val_idx]
            test_target = angles_all[test_idx]
        else:
            labels_tr, labels_val = labels[train_idx][tr_idx], labels[train_idx][val_idx]
            test_target, yidx_test = yidx_test, yidx_test
    else:
        (Xtr, Xval, X_test), (ltr, lval, lte) = train_val_test_split_simple(X, labels, seed=seed + iteration)
        if is_regression:
            labels_tr = np.array([angle_mapping[l] for l in ltr], dtype=np.float32)
            labels_val = np.array([angle_mapping[l] for l in lval], dtype=np.float32)
            test_target = np.array([angle_mapping[l] for l in lte], dtype=np.float32)
            yidx_test = np.array([label_to_idx[l] for l in lte], dtype=np.int32)
        else:
            labels_tr, labels_val = ltr, lval
            test_target = np.array([label_to_idx[l] for l in lte], dtype=np.int32)
            yidx_test = test_target

    return Xtr, Xval, X_test, labels_tr, labels_val, test_target, yidx_test


def run_single_iteration(Xtr, Xval, X_test, labels_tr, labels_val, test_target, yidx_test,
                        is_regression, is_multitask, label_to_idx, angle_mapping, args, fold_name, tolerance, phases):
    """Execute training and evaluation for a single iteration"""
    IMAGE_SIZE, num_classes = 66, len(phases)

    # Prepare targets and model
    if is_multitask:
        # マルチタスク: 回帰と分類の両方
        ytr_circle = angles_2_unit_circle_points(labels_tr)
        yval_circle = angles_2_unit_circle_points(labels_val)
        ytr_idx = np.array([label_to_idx[angle_to_label_with_mapping(a, angle_mapping)] for
a in labels_tr], dtype=np.int32)
        yval_idx = np.array([label_to_idx[angle_to_label_with_mapping(a, angle_mapping)] for
a in labels_val], dtype=np.int32)

        ytr = {'regression': ytr_circle, 'classification': ytr_idx}
        yval = {'regression': yval_circle, 'classification': yval_idx}

        model = create_jurkat_multitask_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1),
num_classes=num_classes)
        model.compile(
            optimizer='adam',
            loss={'regression': cos_similarity_loss, 'classification':
'sparse_categorical_crossentropy'},
            loss_weights={'regression': 1.0, 'classification': 1.0},
            metrics={'classification': ['accuracy']}
        )
        monitor, mode = 'val_loss', 'min'

    elif is_regression:
        ytr, yval = angles_2_unit_circle_points(labels_tr), angles_2_unit_circle_points(labels_val)
        model = create_jurkat_regression_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1))
        model.compile(optimizer='adam', loss=cos_similarity_loss)
        monitor, mode = 'val_loss', 'min'
    else:
        ytr = np.array([label_to_idx[l] for l in labels_tr], dtype=np.int32)
        yval = np.array([label_to_idx[l] for l in labels_val], dtype=np.int32)
        model = create_jurkat_classification_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1), num_classes=num_classes)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        monitor, mode = 'val_accuracy', 'max'

    # Train
    callbacks = []

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=4,
        restore_best_weights=True,
        mode=mode,
        verbose=1
    )
    callbacks.append(early_stop)
    
    if args.save_weights:
        subdir = 'jurkat_7cls' if is_regression else 'jurkat_7cls_cls'
        if is_multitask:
            subdir = 'jurkat_7cls_multitask'
        mc = setup_checkpoint(f'weights/{subdir}', f'{fold_name}.keras', monitor=monitor, mode=mode, save_weights_only=(not is_regression and not is_multitask))
        callbacks.append(mc)
    model.fit(Xtr, ytr, validation_data=(Xval, yval), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=1)

    # Evaluate
    if is_multitask:
        # マルチタスク評価
        preds = model.predict(X_test, verbose=0)
        preds_circle = preds[0]  # 回帰出力
        preds_class = preds[1]   # 分類出力

        # 回帰評価
        preds_circle = associated_points_on_circle(preds_circle)
        pred_angles = points_2_angles(preds_circle)
        mean_dev = prediction_mean_deviation(test_target, pred_angles)
        tol_acc = tolerance_accuracy(pred_angles, test_target, tol=tolerance)

        # 分類評価
        y_pred_idx = np.argmax(preds_class, axis=1)
        from sklearn.metrics import accuracy_score
        cls_acc = accuracy_score(yidx_test, y_pred_idx)

        if args.confmat:
            pred_labels = [angle_to_label_with_mapping(a, angle_mapping) for a in pred_angles]
            y_pred_idx_reg = np.array([label_to_idx[l] for l in pred_labels], dtype=np.int32)
            save_confusion_matrix(yidx_test, y_pred_idx_reg, phases, f'{args.out_dir}/multitask_regression', args.confmat_norm, fold_name)
            save_confusion_matrix(yidx_test, y_pred_idx, phases, f'{args.out_dir}/multitask_classification', args.confmat_norm, fold_name)

        return (mean_dev, cls_acc), tol_acc, model

    elif is_regression:
        preds = associated_points_on_circle(model.predict(X_test, verbose=0))
        pred_angles = points_2_angles(preds)
        result = prediction_mean_deviation(test_target, pred_angles)
        tol_acc = tolerance_accuracy(pred_angles, test_target, tol=tolerance)
        if args.confmat:
            pred_labels = [angle_to_label_with_mapping(a, angle_mapping) for a in pred_angles]
            y_pred_idx = np.array([label_to_idx[l] for l in pred_labels], dtype=np.int32)
            save_confusion_matrix(yidx_test, y_pred_idx, phases, f'{args.out_dir}/regression', args.confmat_norm, fold_name)
        return result, tol_acc, model
    else:
        _, result = model.evaluate(X_test, test_target, verbose=0)
        if args.confmat:
            y_pred_idx = np.argmax(model.predict(X_test, verbose=0), axis=1)
            save_confusion_matrix(test_target, y_pred_idx, phases, f'{args.out_dir}/classification', args.confmat_norm, fold_name)
        return result, None, model

def main():
    parser = argparse.ArgumentParser(description='Jurkat unified training (classification/regression/multitask)')
    parser.add_argument('--task', choices=['classification', 'regression', 'multitask'], required=True,
                       help='Task type: classification (softmax) or regression (circular)')
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--runs', '-r', type=int, default=1)
    parser.add_argument('--folds', type=int, default=1,
                       help='If >1, perform stratified K-fold CV (runs is ignored)')
    parser.add_argument('--limit_per_phase', type=int, default=None,
                       help='Limit samples per phase for quick testing')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--confmat', action='store_true', help='Save confusion matrix')
    parser.add_argument('--confmat_norm', choices=['none', 'true', 'pred', 'all'], default='none')
    parser.add_argument('--out_dir', type=str, default='results/confusion_matrices')
    args = parser.parse_args()

    # Constants
    IMAGE_SIZE = 66  # Original Jurkat image size
    TOLERANCE = 180.0 / 7.0  # Half of inter-class spacing (≈25.7°)

    is_regression = (args.task == 'regression')
    is_multitask = (args.task == 'multitask')
    angle_mapping = build_angle_mapping() if (is_regression or is_multitask) else None

    print(f'=== Jurkat 7-class {args.task} ===')
    print(f'Phases: {PHASES7}')
    print(f'Image size: {IMAGE_SIZE}x{IMAGE_SIZE}')
    if is_regression:
        print(f'Angle mapping: {angle_mapping}')
        print(f'Tolerance: ±{TOLERANCE:.2f}°')

    # Load data
    # Xには画像データ、labelsには対応する位相ラベルが入る
    X, labels = load_jurkat_ch3_data(limit_per_phase=args.limit_per_phase, image_size=IMAGE_SIZE)
    # 移相ラベルを整数に変換
    label_to_idx = get_label_to_index_mapping(PHASES7)
    # 文字列を整数に
    y_all_idx = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

    if is_regression or is_multitask:
        # 角度だけの配列を作成
        angles_all = np.array([angle_mapping[l] for l in labels], dtype=np.float32)

    # Determine iteration mode
    all_results, all_tol_acc = [], []
    use_cv = args.folds and args.folds > 1
    n_iterations = args.folds if use_cv else args.runs

    if use_cv:
        print(f'Using stratified {args.folds}-fold CV')
        # 層化KFoldオブジェクトの作成
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        # (1, (train_idx, test_idx))   Fold 1 
        # (2, (train_idx, test_idx))   Fold 2
        # (3, (train_idx, test_idx))   Fold 3
        # ...のように分割する
        splits = list(enumerate(skf.split(X, y_all_idx), 1))
    else:
        # (1, None),
        # (2, None),
        # ...
        splits = [(i + 1, None) for i in range(args.runs)]

    # Iterate over splits
    for iteration, split_indices in splits:
        fold_name = f'fold{iteration}' if use_cv else f'run{iteration}'
        print(f'\n{"Fold" if use_cv else "Run"} {iteration}/{n_iterations}')

        # Prepare data splits
        # Xtr, Xval, X_test: 訓練/検証/テスト用の画像データ
        # labels_tr, labels_val: 訓練/検証用のラベル（角度 or 文字列）
        # test_target: テスト用のターゲット（角度 or クラスインデックス）
        # yidx_test: テスト用のクラスインデックス（confusion matrix用）
        Xtr, Xval, X_test, labels_tr, labels_val, test_target, yidx_test = prepare_split_data(
            use_cv, iteration, X, labels, y_all_idx, angles_all if (is_regression or is_multitask) else None,
            is_regression or is_multitask, label_to_idx, angle_mapping, args.seed,
            train_idx=split_indices[0] if use_cv else None,
            test_idx=split_indices[1] if use_cv else None
        )
        print(f'Data: train={Xtr.shape}, val={Xval.shape}, test={X_test.shape}')

        # Train and evaluate
        # regressionの場合: result = 平均角度誤差、tol_acc = 許容範囲内正解率
        # classificationの場合: result = 精度、tol_acc = None
        result, tol_acc, model = run_single_iteration(
            Xtr, Xval, X_test, labels_tr, labels_val, test_target, yidx_test,
            is_regression, is_multitask, label_to_idx, angle_mapping, args, fold_name, TOLERANCE, PHASES7
        )

        all_results.append(result)
        if tol_acc is not None:
            all_tol_acc.append(tol_acc)

        # Print results
        if is_multitask:
            mean_dev, cls_acc = result
            print(f'{fold_name}: regression_dev={mean_dev:.2f}°, tol_acc@±{TOLERANCE:.2f}°={tol_acc*100:.2f}%, classification_acc={cls_acc*100:.2f}%')
        elif is_regression:
            print(f'{fold_name}: deviation={result:.2f}°, tol_acc@±{TOLERANCE:.2f}°={tol_acc*100:.2f}%')
        else:
            print(f'{fold_name}: accuracy={result*100:.2f}%')
            if iteration == n_iterations:
                y_pred_idx = np.argmax(model.predict(X_test, verbose=0), axis=1)
                print('\nClassification report:')
                print(classification_report(test_target, y_pred_idx, target_names=PHASES7, digits=4))

    # Summary
    print('\n=== Summary ===')
    n = args.folds if (args.folds and args.folds > 1) else args.runs
    # Print results
    if is_multitask:
        mean_dev, cls_acc = result
        print(f'{fold_name}: regression_dev={mean_dev:.2f}°, tol_acc@±{TOLERANCE:.2f}°={tol_acc*100:.2f}%, classification_acc={cls_acc*100:.2f}%')
    elif is_regression:
        print(f'{fold_name}: deviation={result:.2f}°, tol_acc@±{TOLERANCE:.2f}°={tol_acc*100:.2f}%')
    else:
        print(f'{fold_name}: accuracy={result*100:.2f}%')
        if iteration == n_iterations:
            y_pred_idx = np.argmax(model.predict(X_test, verbose=0), axis=1)
            print('\nClassification report:')
            print(classification_report(test_target, y_pred_idx, target_names=PHASES7, digits=4))

if __name__ == '__main__':
    main()
