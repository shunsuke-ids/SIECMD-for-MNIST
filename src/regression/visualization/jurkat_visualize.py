#!/usr/bin/env python3
"""
Visualization utilities for Jurkat (CellCycle) experiments:

- reg_circle: Plot unit-circle scatter of circular regression predictions on a chosen CV fold
- tsne_cls:   Run t-SNE on the penultimate features of the classification baseline and scatter by class

Assumptions:
- Data loader and model backbone match jurkat_cyclic_regression.py and jurkat_classification_baseline.py
- Checkpoints exist under weights/jurkat_7cls/ (regression, full model saves) and
  weights/jurkat_7cls_cls/ (classification, weights-only)
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np

# plotting
import matplotlib
matplotlib.use('Agg')  # headless save
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# Import project symbols
from src.regression.utils.data_loaders import load_jurkat_ch3_data, get_label_to_index_mapping, PHASES7
from src.regression.utils.model_builders import create_jurkat_classification_model
from src.regression.utils.training_utils import get_cv_fold_indices
from src.regression.utils.format_gt import (
    associated_points_on_circle, points_2_angles, build_angle_mapping_equal
)
from src.DL.activation_functions import sigmoid_activation
from src.DL.losses import linear_dist_squared_loss

from keras import models as km

ANGLE_STEP_7 = 360.0 / len(PHASES7)

def build_angle_mapping():
    return build_angle_mapping_equal(PHASES7, start_angle=0)


def plot_regression_circle(args):
    # Load data
    X, labels = load_jurkat_ch3_data(limit_per_phase=args.limit_per_phase, image_size=args.image_size)
    label_to_idx = get_label_to_index_mapping(PHASES7)
    y_all_idx = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

    # Indices for target fold
    tr_idx, val_idx, te_idx, _ = get_cv_fold_indices(y_all_idx, n_splits=args.folds, target_fold=args.fold, seed=args.seed)
    X_test = X[te_idx]
    y_test_idx = y_all_idx[te_idx]

    # Load trained regression model (full model saved)
    ckpt_path = Path('weights/jurkat_7cls') / f'fold{args.fold}.keras'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Regression checkpoint not found: {ckpt_path}. Train with CV first.')
    model = km.load_model(str(ckpt_path), custom_objects={
        'sigmoid_activation': sigmoid_activation,
        'linear_dist_squared_loss': linear_dist_squared_loss,
    })

    # Predict and project to unit circle
    preds = model.predict(X_test, verbose=0)
    preds = associated_points_on_circle(preds)
    pred_angles = points_2_angles(preds)

    # Build class centers on unit circle
    angle_map = build_angle_mapping()
    centers = []
    for ph in PHASES7:
        ang = np.deg2rad(angle_map[ph])
        centers.append([np.cos(ang), np.sin(ang)])
    centers = np.array(centers)

    # Scatter plot on unit circle
    colors = plt.cm.tab10(np.linspace(0, 1, len(PHASES7)))
    fig, ax = plt.subplots(figsize=(6, 6))
    # unit circle
    t = np.linspace(0, 2*np.pi, 512)
    ax.plot(np.cos(t), np.sin(t), 'k-', lw=1, alpha=0.3)

    for i, ph in enumerate(PHASES7):
        mask = (y_test_idx == i)
        pts = preds[mask]
        if pts.size == 0:
            continue
        ax.scatter(pts[:,0], pts[:,1], s=6, color=colors[i], alpha=0.5, label=ph)

    # centers
    ax.scatter(centers[:,0], centers[:,1], c=colors[:len(PHASES7)], s=80, marker='x', lw=2, label='centers')
    for i, ph in enumerate(PHASES7):
        ax.text(centers[i,0]*1.08, centers[i,1]*1.08, ph, color=colors[i], ha='center', va='center', fontsize=8)

    ax.set_aspect('equal'); ax.set_xlim([-1.2,1.2]); ax.set_ylim([-1.2,1.2])
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'Circular regression predictions (fold {args.fold})')
    ax.legend(loc='upper right', fontsize=8, ncol=2, frameon=False)

    out_dir = Path('figs'); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'reg_circle_fold{args.fold}.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved: {out_path}')


def plot_tsne_classification(args):
    # Load data
    X, labels = load_jurkat_ch3_data(limit_per_phase=args.limit_per_phase, image_size=args.image_size)
    label_to_idx = get_label_to_index_mapping(PHASES7)
    y_all_idx = np.array([label_to_idx[l] for l in labels], dtype=np.int32)

    # Indices for target fold
    tr_idx, val_idx, te_idx, _ = get_cv_fold_indices(y_all_idx, n_splits=args.folds, target_fold=args.fold, seed=args.seed)
    X_test = X[te_idx]
    y_test_idx = y_all_idx[te_idx]

    # Build classification model and load weights
    model = create_jurkat_classification_model(input_shape=(args.image_size, args.image_size, 1), num_classes=len(PHASES7))
    weights_path = Path('weights/jurkat_7cls_cls') / f'fold{args.fold}.keras'
    if not weights_path.exists():
        raise FileNotFoundError(f'Classification weights not found: {weights_path}. Train with --save_weights and CV first.')
    model.load_weights(str(weights_path))

    # Feature extractor (penultimate layer)
    feat_model = km.Model(model.input, model.get_layer('penultimate_features').output)
    feats = feat_model.predict(X_test, verbose=0)

    # Optional subsample for speed
    if args.sample is not None and args.sample < feats.shape[0]:
        rng = np.random.default_rng(args.seed)
        sel = rng.choice(feats.shape[0], size=args.sample, replace=False)
        feats = feats[sel]
        y_test_idx = y_test_idx[sel]

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=args.perplexity, init='pca', learning_rate='auto', random_state=args.seed)
    Z = tsne.fit_transform(feats)

    colors = plt.cm.tab10(np.linspace(0, 1, len(PHASES7)))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, ph in enumerate(PHASES7):
        mask = (y_test_idx == i)
        if not np.any(mask):
            continue
        ax.scatter(Z[mask,0], Z[mask,1], s=8, color=colors[i], alpha=0.7, label=ph)
    ax.set_title(f't-SNE of classification features (fold {args.fold})')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='best', fontsize=8, ncol=2, frameon=False)

    out_dir = Path('figs'); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'tsne_cls_fold{args.fold}.png'
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f'Saved: {out_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualization for Jurkat regression (circle) and classification (t-SNE).')
    parser.add_argument('--mode', choices=['reg_circle', 'tsne_cls'], required=True)
    parser.add_argument('--fold', type=int, default=1, help='Target CV fold number (1-based).')
    parser.add_argument('--folds', type=int, default=5, help='Total number of folds used in training.')
    parser.add_argument('--image_size', type=int, default=66)
    parser.add_argument('--limit_per_phase', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    # t-SNE specific
    parser.add_argument('--perplexity', type=float, default=30.0)
    parser.add_argument('--sample', type=int, default=None, help='Optional subsample size for t-SNE.')
    args = parser.parse_args()

    if args.mode == 'reg_circle':
        plot_regression_circle(args)
    elif args.mode == 'tsne_cls':
        plot_tsne_classification(args)
    else:
        raise ValueError('Unknown mode')


if __name__ == '__main__':
    main()
