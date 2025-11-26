"""
データセット：phenocam, jurkat
モデル：classification, regression, multitask
使用方法：
    # PHENOCAM classification with 5-folds
    python src/regression/training/unified_train.py --dataset phenocam --task classification --epochs 30 --folds 5

    # Jurkat regression
    python src/regression/training/unified_train.py --dataset jurkat --task regression --epochs 20 --folds 5
      
    # PHENOCAM multi-task
    python src/regression/training/unified_train.py --dataset phenocam --task multitask --epochs 30 --folds 5
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from keras.callbacks import EarlyStopping

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.append(project_root)

from src.DL.metrics import prediction_mean_deviation
from src.DL.losses import cos_similarity_loss
from src.regression.utils.format_gt import (
    points_2_angles, associated_points_on_circle, angles_2_unit_circle_points,
    build_angle_mapping_equal, angle_to_label_with_mapping
)
from src.regression.utils.data_loaders import (
    load_jurkat_ch3_data, load_phenocam_seasonal_data,
    get_label_to_index_mapping, PHASES7, SEASONS
)
from src.regression.utils.model_builders import (
    create_jurkat_classification_model, create_jurkat_regression_model, create_jurkat_multitask_model,
    create_phenocam_classification_model, create_phenocam_regression_model, create_phenocam_multitask_model
)
from src.regression.utils.training_utils import setup_checkpoint, save_confusion_matrix, tolerance_accuracy

DATASET_CONFIGS = {
      'jurkat': {
          'labels': PHASES7,
          'image_size': 66,
          'num_classes': 7,
          'channels': 1,
          'loader': load_jurkat_ch3_data,
          'loader_param': 'limit_per_phase',
          'model_builders': {
              'classification': create_jurkat_classification_model,
              'regression': create_jurkat_regression_model,
              'multitask': create_jurkat_multitask_model
          },
          'tolerance': 180.0 / 7.0,  # Half of inter-class spacing (≈25.7°)
          'weights_dir_suffix': '7cls',
          'batch_size': 64,
          'patience': 6
      },
      'phenocam': {
          'labels': SEASONS,
          'image_size': 224,
          'num_classes': 4,
          'channels': 3,
          'loader': load_phenocam_seasonal_data,
          'loader_param': 'limit_per_season',
          'model_builders': {
              'classification': create_phenocam_classification_model,
              'regression': create_phenocam_regression_model,
              'multitask': create_phenocam_multitask_model
          },
          'tolerance': 180.0 / 4.0,  # Half of inter-class spacing (45°)
          'weights_dir_suffix': '4seasons',
          'batch_size': 64,  # Reduced from 32 to prevent OOM
          'patience': 6  # Increased from 5 to allow more training
      }    
}

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
    """Prepare train/val/test splits based on CV or simple split"""
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
                         is_regression, is_multitask, label_to_idx, angle_mapping, args, config, fold_name):
    IMAGE_SIZE = config['image_size']
    CHANNELS = config['channels']
    num_classes = config['num_classes']
    labels_list = config['labels']
    tolerance = config['tolerance']

    if is_multitask:
        ytr_circle = angles_2_unit_circle_points(labels_tr)
        yval_circle = angles_2_unit_circle_points(labels_val)
        ytr_idx = np.array([label_to_idx[angle_to_label_with_mapping(angle, angle_mapping)] for angle in labels_tr], dtype=np.int32)
        yval_idx = np.array([label_to_idx[angle_to_label_with_mapping(angle, angle_mapping)] for angle in labels_val], dtype=np.int32)

        ytr = {'regression': ytr_circle, 'classification': ytr_idx}
        yval = {'regression': yval_circle, 'classification': yval_idx}

        model = config['model_builders']['multitask'](
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            num_classes=num_classes,
        )
        model.compile(
            optimizer='adam',
            loss={
                'regression': cos_similarity_loss,
                'classification': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'regression': 1.0,
                'classification': 1.0
            },
            metrics={
                'classification': ['accuracy']
            }
        )
        monitor, mode = 'val_loss', 'min'

    elif is_regression:
        ytr = angles_2_unit_circle_points(labels_tr)
        yval = angles_2_unit_circle_points(labels_val)

        model = config['model_builders']['regression'](
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
        )
        model.compile(
            optimizer='adam',
            loss=cos_similarity_loss,
        )
        monitor, mode = 'val_loss', 'min'

    else:
        ytr = np.array([label_to_idx[l] for l in labels_tr], dtype=np.int32)
        yval = np.array([label_to_idx[l] for l in labels_val], dtype=np.int32)

        model = config['model_builders']['classification'](
            input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS),
            num_classes=num_classes,
        )
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        monitor, mode = 'val_accuracy', 'max'

    callbacks = []

    early_stop = EarlyStopping(
        monitor=monitor,
        patience=config['patience'],
        restore_best_weights=True,
        mode=mode,
        verbose=1
    )
    callbacks.append(early_stop)

    if args.save_weights:
        if is_multitask:
            subdir = f"{args.dataset}_{config['weights_dir_suffix']}_multitask"
        elif is_regression:
            subdir = f"{args.dataset}_{config['weights_dir_suffix']}_regression"
        else:
            subdir = f"{args.dataset}_{config['weights_dir_suffix']}_classification"

        mc = setup_checkpoint(
            f'weights/{subdir}',
            f'{fold_name}.keras',
            monitor=monitor,
            mode=mode,
            save_weights_only=(not is_regression and not is_multitask)
        )
        callbacks.append(mc)
    
    model.fit(Xtr, ytr, validation_data=(Xval, yval),
              epochs=args.epochs, batch_size=args.batch_size,
              callbacks=callbacks, verbose=1)
    
    if is_multitask:
        preds = model.predict(X_test, batch_size=8, verbose=0)
        preds_circle = preds[0]
        preds_class = preds[1]

        preds_circle = associated_points_on_circle(preds_circle)
        pred_angles = points_2_angles(preds_circle)
        mean_dev = prediction_mean_deviation(test_target, pred_angles)
        tol_acc = tolerance_accuracy(pred_angles, test_target, tolerance)

        y_pred_idx = np.argmax(preds_class, axis=1)
        cls_acc = accuracy_score(yidx_test, y_pred_idx)

        if args.confmat:
            pred_labels = [angle_to_label_with_mapping(angle, angle_mapping) for angle in pred_angles]
            y_pred_idx_reg = np.array([label_to_idx[l] for l in pred_labels], dtype=np.int32)
            save_confusion_matrix(yidx_test, y_pred_idx_reg, labels_list,
                                  f'{args.out_dir}/multitask_regression', args.confmat_norm, fold_name)
            save_confusion_matrix(yidx_test, y_pred_idx, labels_list,
                                  f'{args.out_dir}/multitask_classification', args.confmat_norm, fold_name)
            
        return (mean_dev, cls_acc), tol_acc, model
    
    elif is_regression:
        preds = associated_points_on_circle(model.predict(X_test, batch_size=8, verbose=0))
        pred_angles = points_2_angles(preds)
        result = prediction_mean_deviation(test_target, pred_angles)
        tol_acc = tolerance_accuracy(pred_angles, test_target, tol=tolerance)

        if args.confmat:
              pred_labels = [angle_to_label_with_mapping(a, angle_mapping) for a in pred_angles]
              y_pred_idx = np.array([label_to_idx[l] for l in pred_labels], dtype=np.int32)
              save_confusion_matrix(yidx_test, y_pred_idx, labels_list,
                                  f'{args.out_dir}/regression', args.confmat_norm, fold_name)
        return result, tol_acc, model
    else:
        _, result = model.evaluate(X_test, test_target, verbose=0)

        if args.confmat:
            y_pred_idx = np.argmax(model.predict(X_test, batch_size=8, verbose=0), axis=1)
            save_confusion_matrix(test_target, y_pred_idx, labels_list,
                                  f'{args.out_dir}/classification', args.confmat_norm, fold_name)
        return result, None, model


def main():
    parser = argparse.ArgumentParser(description='Unified training script for multiple datasets')
    parser.add_argument('--dataset', choices=['jurkat', 'phenocam'], required=True,
                        help='Dataset to use')
    parser.add_argument('--task', choices=['classification', 'regression', 'multitask'], required=True,
                        help='Task type')
    parser.add_argument('--epochs', '-e', type=int, default=20)
    parser.add_argument('--batch_size', '-b', type=int, default=None,
                        help='Batch size (default: dataset-specific)')
    parser.add_argument('--runs', '-r', type=int, default=1)
    parser.add_argument('--folds', type=int, default=1,
                        help='If >1, perform stratified K-fold CV (runs is ignored)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit samples per class for quick testing')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--confmat', action='store_true', help='Save confusion matrix')
    parser.add_argument('--confmat_norm', choices=['none', 'true', 'pred', 'all'], default='none')
    parser.add_argument('--out_dir', type=str, default='results/confusion_matrices')
    args = parser.parse_args()

    # Get dataset configuration
    config = DATASET_CONFIGS[args.dataset]

    # Use dataset-specific batch size if not specified
    if args.batch_size is None:
        args.batch_size = config['batch_size']

    # Setup
    is_regression = (args.task == 'regression')
    is_multitask = (args.task == 'multitask')
    angle_mapping = build_angle_mapping_equal(config['labels'], start_angle=0) if (is_regression or is_multitask) else None

    print(f"=== {args.dataset.upper()} {config['num_classes']}-class {args.task} ===")
    print(f"Classes: {config['labels']}")
    print(f"Image size: {config['image_size']}x{config['image_size']} (channels={config['channels']})")
    if is_regression or is_multitask:
        print(f"Angle mapping: {angle_mapping}")
        print(f"Tolerance: ±{config['tolerance']:.2f}°")

    # Load data
    print(f"Loading {args.dataset} data...")
    loader_kwargs = {
        config['loader_param']: args.limit,
        'image_size': config['image_size']
    }
    X, labels = config['loader'](**loader_kwargs)
    print(f"Loaded {X.shape[0]} images")

    label_to_idx = get_label_to_index_mapping(config['labels'])
    y_all_idx = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

    if is_regression or is_multitask:
        angles_all = np.array([angle_mapping[l] for l in labels], dtype=np.float32)

    # Determine iteration mode
    all_results, all_tol_acc = [], []
    use_cv = args.folds and args.folds > 1
    n_iterations = args.folds if use_cv else args.runs

    if use_cv:
        print(f"Using stratified {args.folds}-fold CV")
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = list(enumerate(skf.split(X, y_all_idx), 1))
    else:
        splits = [(i + 1, None) for i in range(args.runs)]

    # Iterate over splits
    for iteration, split_indices in splits:
        fold_name = f'fold{iteration}' if use_cv else f'run{iteration}'
        print(f'\n{"Fold" if use_cv else "Run"} {iteration}/{n_iterations}')

        # Prepare data splits
        Xtr, Xval, X_test, labels_tr, labels_val, test_target, yidx_test = prepare_split_data(
            use_cv, iteration, X, labels, y_all_idx,
            angles_all if (is_regression or is_multitask) else None,
            is_regression or is_multitask, label_to_idx, angle_mapping, args.seed,
            train_idx=split_indices[0] if use_cv else None,
            test_idx=split_indices[1] if use_cv else None
        )
        print(f"Data: train={Xtr.shape}, val={Xval.shape}, test={X_test.shape}")

        # Debug: Check data distribution
        if is_regression or is_multitask:
            train_labels = [angle_to_label_with_mapping(a, angle_mapping) for a in labels_tr]
            val_labels = [angle_to_label_with_mapping(a, angle_mapping) for a in labels_val]
            test_labels_str = [config['labels'][idx] for idx in yidx_test]
            print(f"Train distribution: {dict(sorted(Counter(train_labels).items()))}")
            print(f"Val distribution: {dict(sorted(Counter(val_labels).items()))}")
            print(f"Test distribution: {dict(sorted(Counter(test_labels_str).items()))}")
        else:
            print(f"Train distribution: {dict(sorted(Counter(labels_tr).items()))}")
            print(f"Val distribution: {dict(sorted(Counter(labels_val).items()))}")
            test_labels_str = [config['labels'][idx] for idx in yidx_test]
            print(f"Test distribution: {dict(sorted(Counter(test_labels_str).items()))}")

        # Train and evaluate
        result, tol_acc, model = run_single_iteration(
            Xtr, Xval, X_test, labels_tr, labels_val, test_target, yidx_test,
            is_regression, is_multitask, label_to_idx, angle_mapping, args, config, fold_name
        )

        all_results.append(result)
        if tol_acc is not None:
            all_tol_acc.append(tol_acc)

        # Print results
        if is_multitask:
            mean_dev, cls_acc = result
            print(f"{fold_name}: regression_dev={mean_dev:.2f}°, "
                f"tol_acc@±{config['tolerance']:.2f}°={tol_acc*100:.2f}%, "
                f"classification_acc={cls_acc*100:.2f}%")
        elif is_regression:
            print(f"{fold_name}: deviation={result:.2f}°, "
                f"tol_acc@±{config['tolerance']:.2f}°={tol_acc*100:.2f}%")
        else:
            print(f"{fold_name}: accuracy={result*100:.2f}%")
            if iteration == n_iterations:
                y_pred_idx = np.argmax(model.predict(X_test, batch_size=8, verbose=0), axis=1)
                print('\nClassification report:')
                print(classification_report(test_target, y_pred_idx,
                                        target_names=config['labels'], digits=4))

    # Summary
    print('\n=== Summary ===')
    if is_multitask:
        mean_devs = [r[0] for r in all_results]
        cls_accs = [r[1] for r in all_results]
        print(f"Mean regression deviation: {np.mean(mean_devs):.2f}° ± {np.std(mean_devs):.2f}°")
        print(f"Mean tolerance accuracy: {np.mean(all_tol_acc)*100:.2f}% ± {np.std(all_tol_acc)*100:.2f}%")
        print(f"Mean classification accuracy: {np.mean(cls_accs)*100:.2f}% ± {np.std(cls_accs)*100:.2f}%")
    elif is_regression:
        print(f"Mean deviation: {np.mean(all_results):.2f}° ± {np.std(all_results):.2f}°")
        print(f"Mean tolerance accuracy: {np.mean(all_tol_acc)*100:.2f}% ± {np.std(all_tol_acc)*100:.2f}%")
    else:
        print(f"Mean accuracy: {np.mean(all_results)*100:.2f}% ± {np.std(all_results)*100:.2f}%")


if __name__ == '__main__':
    main()        