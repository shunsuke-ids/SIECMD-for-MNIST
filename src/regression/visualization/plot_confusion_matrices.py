#!/usr/bin/env python3
"""
Plot confusion matrix CSV files under a directory tree and save PNG heatmaps.
Usage:
    python plot_confusion_matrices.py --root results/confmats --out_dir figs/confmats

It will walk the root directory, find all 'confusion_matrix.csv' files and produce a PNG next to each CSV (or in --out_dir preserving subfolders).
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

PHASES7 = ['G1','S','G2','Prophase','Metaphase','Anaphase','Telophase']


def plot_cm(cm: np.ndarray, labels, outpath: Path, title: str = None):
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks and label them with the respective list
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='True label', xlabel='Predicted label')
    if title:
        ax.set_title(title)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell
    fmt = '.2f' if np.max(cm) <= 1.0 else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            if fmt == 'd':
                txt = f"{int(val)}"
            else:
                txt = f"{val:.2f}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="black" if val < (np.max(cm) / 2.) else "white")

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def find_and_plot(root: Path, out_root: Path):
    files = list(root.rglob('confusion_matrix.csv'))
    if not files:
        print(f'No confusion_matrix.csv found under {root}')
        return
    for f in files:
        try:
            cm = np.loadtxt(f, delimiter=',')
        except Exception as e:
            print(f'Failed to read {f}: {e}')
            continue
        rel = f.relative_to(root)
        outpath = out_root / rel.parent / (f.stem + '.png')
        # Build a descriptive title; prefix with [regression] if path includes 'regression'
        path_str = str(rel.parent)
        parts_lower = {p.lower() for p in rel.parts}
        prefix = '[regression] ' if 'regression' in parts_lower else ''
        title = prefix + (f.stem + ' — ' + path_str)
        # if 1D or malformed, try to reshape
        if cm.ndim == 1:
            # try square
            n = int(np.sqrt(cm.size))
            if n*n == cm.size:
                cm = cm.reshape((n,n))
        # ensure square
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            print(f'Skipping {f}: unexpected shape {cm.shape}')
            continue
        plot_cm(cm, PHASES7[:cm.shape[0]], outpath, title=title)
        print(f'Wrote {outpath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='results/confmats', help='Root directory to search for confusion_matrix.csv')
    parser.add_argument('--out_dir', type=str, default='figs/confmats', help='Directory to write PNGs (preserves relative paths)')
    args = parser.parse_args()

    find_and_plot(Path(args.root), Path(args.out_dir))
