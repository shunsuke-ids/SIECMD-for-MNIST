#!/usr/bin/env python3
"""
Unified data loading utilities for Jurkat and MNIST experiments.

This module provides reusable functions to load and preprocess datasets,
as well as train/val/test splitting with optional stratification.
"""
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, List, Dict


PHASE_DIR = Path('/home/shunsuke/data/raw/extracted/CellCycle')
PHASES7 = ['G1', 'S', 'G2', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']


def load_jurkat_ch3_data(limit_per_phase: int = None, image_size: int = 66) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load Jurkat (CellCycle) brightfield (Ch3) images for 7 phases.

    Args:
        limit_per_phase: Optional cap per phase for quick prototyping
        image_size: Side length for resize (square). Default 66 (original dataset size)

    Returns:
        X: Image array (N, H, W, 1)
        labels: Phase labels (N,)
    """
    X: List[np.ndarray] = []
    phase_labels: List[str] = []
    phase_counts: Dict[str, int] = {p: 0 for p in PHASES7}

    # Brightfield Ch3 JPEG files
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
                uniq_files.append(p)
                seen.add(p)

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


def load_mnist_data():
    """
    Load MNIST dataset and normalize.

    Returns:
        (x_train, y_train): Training data and labels
        (x_test, y_test): Test data and labels
    """
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Add channel dimension (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    return (x_train, y_train), (x_test, y_test)


def train_val_test_split(X, y, labels=None, val_ratio=0.15, test_ratio=0.15, seed=42, stratify=False):
    """
    Split data into train/val/test sets.

    Args:
        X: Feature array
        y: Target array (angles, class indices, etc.)
        labels: Optional label array for stratification (e.g., phase labels)
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        stratify: If True and labels is provided, use stratified split

    Returns:
        (X_train, X_val, X_test): Split features
        (y_train, y_val, y_test): Split targets
        (labels_train, labels_val, labels_test): Split labels (if labels provided, else None)
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idx = np.arange(N)

    if not stratify or labels is None:
        # Simple random split
        rng.shuffle(idx)
        test_size = int(N * test_ratio)
        val_size = int(N * val_ratio)
        test_idx = idx[:test_size]
        val_idx = idx[test_size:test_size + val_size]
        train_idx = idx[test_size + val_size:]
    else:
        # Stratified split not implemented in this basic version
        # For stratified splits, use sklearn's StratifiedShuffleSplit directly in calling code
        raise NotImplementedError("Stratified split should use sklearn directly in training scripts")

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    if labels is not None:
        labels_train = labels[train_idx]
        labels_val = labels[val_idx]
        labels_test = labels[test_idx]
        return (X_train, X_val, X_test), (y_train, y_val, y_test), (labels_train, labels_val, labels_test)
    else:
        return (X_train, X_val, X_test), (y_train, y_val, y_test)


def prepare_mnist_for_regression(similarity_based=False):
    """
    Prepare MNIST data for circular regression.

    Args:
        similarity_based: If True, use visual similarity-based angle mapping

    Returns:
        (x_train, x_val, x_test): Split features
        (y_train_circular, y_val_circular, y_test_circular): Circular coordinates
        angles_test: Test set angles for evaluation
    """
    from src.regression.utils.format_gt import angles_2_unit_circle_points
    from sklearn.model_selection import train_test_split

    (x_train, y_train), (x_test, y_test) = load_mnist_data()

    # Angle mapping
    if similarity_based:
        angle_mapping = {
            0: 0, 6: 36, 9: 72, 8: 108, 3: 144,
            2: 180, 5: 216, 1: 252, 7: 288, 4: 324
        }
    else:
        angle_mapping = {i: i * 36.0 for i in range(10)}

    # Convert labels to angles
    angles_train = np.array([angle_mapping[label] for label in y_train])
    angles_test = np.array([angle_mapping[label] for label in y_test])

    # Convert angles to unit circle points
    y_train_circular = angles_2_unit_circle_points(angles_train)
    y_test_circular = angles_2_unit_circle_points(angles_test)

    # Train/val split
    x_train, x_val, y_train_circular, y_val_circular = train_test_split(
        x_train, y_train_circular, test_size=0.2, random_state=42
    )

    return (x_train, x_val, x_test), (y_train_circular, y_val_circular, y_test_circular), angles_test


def get_label_to_index_mapping(labels=PHASES7):
    """
    Get mapping from label strings to integer indices.

    Args:
        labels: List of label strings

    Returns:
        Dictionary mapping labels to indices
    """
    return {label: i for i, label in enumerate(labels)}

PHENOCAM_DIR = Path('/home/shunsuke/data/raw/phenocam/phenocamdata/ashburnham')
SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']

def load_phenocam_seasonal_data(limit_per_season:int = None, image_size: int = 224):
    """
    Phenocam 画像を季節ごとに読み込む。
    Args:
        image_size: 画像のリサイズ後の一辺の長さ（正方形）。デフォルトは224。
        limit_per_season: 各季節ごとの画像数の上限（プロトタイピング用）。デフォルトはNone（制限なし）。
    
    Returns:
        X: 画像の配列 (N, H, W, 3)
        labels: 季節ラベルの配列 (N,)
    """
    X: List[np.ndarray] = []
    season_labels: List[str] = []
    season_counts: Dict[str, int] = {s: 0 for s in SEASONS}

    for year_dir in sorted(PHENOCAM_DIR.glob('[0-9]*')):
        if not year_dir.is_dir():
            continue

        for month_dir in sorted(year_dir.glob('[0-9]*')):
            if not month_dir.is_dir():
                continue

            month = int(month_dir.name)
            season = month_to_season(month)

            if limit_per_season is not None and season_counts[season] >= limit_per_season:
                continue

            for img_path in sorted(month_dir.glob('*.jpg')):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                if img.shape[0] != image_size or img.shape[1] != image_size:
                    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA) # 縮小にはINTER_AREAを使用するのが推奨らしい

                # 学習しやすいように[0, 255]から[0, 1]に正規化
                img = img.astype(np.float32) / 255.0

                X.append(img)
                season_labels.append(season)
                season_counts[season] += 1

    X = np.stack(X, axis=0)
    labels = np.array(season_labels)

    return X, labels

def month_to_season(month: int) -> str:
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'