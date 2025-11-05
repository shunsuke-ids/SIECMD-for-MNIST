#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import json

def find_fold_confmats(root: Path):
    files = []
    for k in range(1, 100):
        p = root / f'fold{k}' / 'confusion_matrix.csv'
        if p.exists():
            files.append(p)
    if not files:
        files = list(root.rglob('confusion_matrix.csv'))
    return files

def load_and_sum(files):
    cm_sum = None
    for p in files:
        cm = np.loadtxt(p, delimiter=',')
        if cm_sum is None:
            cm_sum = np.array(cm, dtype=np.float64)
        else:
            if cm_sum.shape != cm.shape:
                raise ValueError(f"Shape mismatch: {p} has {cm.shape}, expected {cm_sum.shape}")
            cm_sum += cm
    if cm_sum is None:
        raise FileNotFoundError('No confusion_matrix.csv found')
    return cm_sum

def normalize_cm(cm: np.ndarray, mode: str):
    if mode == 'none':
        return cm.copy()
    if mode == 'all':
        s = cm.sum()
        return cm / s if s > 0 else cm.copy()
    if mode == 'true':
        out = cm.astype(np.float64).copy()
        rs = out.sum(axis=1, keepdims=True)
        mask = rs > 0
        out[mask[:, 0]] = out[mask[:, 0]] / rs[mask]
        return out
    if mode == 'pred':
        out = cm.astype(np.float64).copy()
        cs = out.sum(axis=0, keepdims=True)
        mask = cs > 0
        out[:, mask[0]] = out[:, mask[0]] / cs[:, mask[0]]
        return out
    raise ValueError(f'Unknown normalize mode: {mode}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--normalize', choices=['none','true','pred','all'], default='none')
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_fold_confmats(root)
    if not files:
        raise SystemExit(f'No confusion_matrix.csv under {root}')

    cm_sum = load_and_sum(files)

    raw_path = out_dir / 'confusion_matrix_raw.csv'
    np.savetxt(raw_path, cm_sum, delimiter=',', fmt='%0.6f')

    cm_out = normalize_cm(cm_sum, args.normalize)
    out_path = out_dir / 'confusion_matrix.csv'
    np.savetxt(out_path, cm_out, delimiter=',', fmt='%0.6f')

    total = float(cm_sum.sum())
    correct = float(np.trace(cm_sum))
    errors = float(total - correct)
    acc = (correct / total) if total > 0 else 0.0
    meta = {
        'root': str(root),
        'fold_files': [str(p) for p in files],
        'total': total,
        'correct': correct,
        'errors': errors,
        'accuracy': acc,
        'normalize': args.normalize,
        'raw_csv': str(raw_path),
        'csv': str(out_path),
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(meta, f, indent=2)

if __name__ == '__main__':
    main()