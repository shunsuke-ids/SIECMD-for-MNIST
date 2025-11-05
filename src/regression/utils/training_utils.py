#!/usr/bin/env python3
"""
Training utilities for both Jurkat and MNIST experiments.

This module provides reusable functions for common training tasks such as
confusion matrix saving, checkpoint setup, CV fold splitting, etc.
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
from keras.callbacks import ModelCheckpoint

try:
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    from sklearn.metrics import confusion_matrix
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def save_confusion_matrix(y_true, y_pred, labels, output_dir, normalize='none', fold_name='fold'):
    """
    Compute and save confusion matrix to CSV.

    Args:
        y_true: True labels (integer indices)
        y_pred: Predicted labels (integer indices)
        labels: List of class labels (for determining matrix size)
        output_dir: Output directory path
        normalize: Normalization mode ('none', 'true', 'pred', 'all')
        fold_name: Name for the fold/run (e.g., 'fold1', 'run3')
    """
    if not _HAS_SKLEARN:
        print("scikit-learn not available, skipping confusion matrix save")
        return

    norm = None if normalize == 'none' else normalize
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))), normalize=norm)

    out_dir = Path(output_dir) / fold_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / 'confusion_matrix.csv', cm, delimiter=',', fmt='%0.6f')
    print(f'Confusion matrix saved to {out_dir / "confusion_matrix.csv"}')


def setup_checkpoint(weights_dir, filename, monitor='val_loss', mode='min', save_weights_only=True, verbose=1):
    """
    Create ModelCheckpoint callback with standard settings.

    Args:
        weights_dir: Directory to save weights
        filename: Checkpoint filename (e.g., 'fold1.keras')
        monitor: Metric to monitor
        mode: 'min' or 'max'
        save_weights_only: If True, save only weights; if False, save full model
        verbose: Verbosity level

    Returns:
        ModelCheckpoint callback
    """
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / filename

    return ModelCheckpoint(
        str(checkpoint_path),
        save_best_only=True,
        monitor=monitor,
        mode=mode,
        save_weights_only=save_weights_only,
        verbose=verbose
    )


def get_cv_fold_indices(y_idx: np.ndarray, n_splits: int, target_fold: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recreate the CV split used in training and return train/val/test indices for the target fold.

    Args:
        y_idx: Label indices for stratification
        n_splits: Total number of folds
        target_fold: Target fold number (1-based)
        seed: Random seed

    Returns:
        tr_idx: Training indices (within fold)
        val_idx: Validation indices (within fold)
        te_idx: Test indices (held-out fold)
        tr_idx_full: Full training indices before val split
    """
    if not _HAS_SKLEARN:
        raise RuntimeError("scikit-learn is required for CV fold splitting")

    assert 1 <= target_fold <= n_splits, 'fold must be within [1, n_splits]'

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_num = 0

    for tr_idx_full, te_idx in skf.split(np.zeros_like(y_idx), y_idx):
        fold_num += 1
        if fold_num == target_fold:
            # Derive stratified validation from training fold (15%)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed + fold_num)
            y_tr_full = y_idx[tr_idx_full]
            tr_sub, val_sub = next(sss.split(np.zeros_like(y_tr_full), y_tr_full))
            tr_idx = tr_idx_full[tr_sub]
            val_idx = tr_idx_full[val_sub]
            return tr_idx, val_idx, te_idx, tr_idx_full

    raise RuntimeError('Failed to find requested fold split')


def build_angle_mapping_equal(labels, start_angle=0):
    """
    Build equal-spaced angle mapping for circular regression.

    Args:
        labels: List of class labels
        start_angle: Starting angle (default 0)

    Returns:
        Dictionary mapping labels to angles
    """
    n_classes = len(labels)
    angle_step = 360.0 / n_classes
    return {label: (i * angle_step + start_angle) % 360.0 for i, label in enumerate(labels)}


def angle_to_label(angle: float, angle_mapping: Dict[str, float]) -> str:
    """
    Find the closest label for a given angle using circular distance.

    Args:
        angle: Predicted angle (0-360)
        angle_mapping: Dictionary mapping labels to angles

    Returns:
        Closest label
    """
    best = None
    best_dist = 999.0

    for lbl, ang in angle_mapping.items():
        diff = abs(angle - ang)
        diff = min(diff, 360.0 - diff)  # Circular distance
        if diff < best_dist:
            best_dist = diff
            best = lbl

    return best


def tolerance_accuracy(pred_angles: np.ndarray, true_angles: np.ndarray, tol: float) -> float:
    """
    Calculate tolerance-based accuracy for circular regression.

    Args:
        pred_angles: Predicted angles
        true_angles: True angles
        tol: Tolerance in degrees

    Returns:
        Accuracy (fraction of predictions within tolerance)
    """
    diffs = np.abs(pred_angles - true_angles)
    diffs = np.minimum(diffs, 360 - diffs)  # Circular distance
    return float(np.mean(diffs <= tol))
