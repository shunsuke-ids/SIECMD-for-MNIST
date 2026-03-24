"""
保存済みモデルを使ってソフト混同行列を生成するスクリプト

使い方:
    python pytorch/evaluate_saved.py \
        --dataset jurkat7 \
        --loss ce \
        --model_path pytorch/saved_models/jurkat7_ce_100ep_best.pth \
        --output_dir pytorch/eval_results
"""
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from train import DATASETS, LOSS_FUNCTIONS, VECTOR_LOSSES, evaluate_detailed, set_seed
from models import SimpleCNN, VonMisesModel
from data_loaders import (get_mnist_loaders, get_jurkat_loaders,
                      get_sysmex_loaders, get_sysmex_7class_loaders,
                      get_phenocam_loaders)
from losses import (SoftmaxVectorLoss, NormalizedSoftmaxVectorLoss,
                    MSEVectorLoss, EuclideanVectorLoss, ArcDistanceVectorLoss)


def get_test_loader(dataset_name, batch_size=64, seed=42):
    """データセット名からテストローダーを取得（末尾がtest_loader）"""
    if dataset_name == 'mnist':
        _, test_loader = get_mnist_loaders(batch_size)
    elif dataset_name == 'jurkat':
        _, _, test_loader = get_jurkat_loaders(batch_size, num_classes=3)
    elif dataset_name == 'jurkat4':
        _, _, test_loader = get_jurkat_loaders(batch_size, num_classes=4)
    elif dataset_name == 'jurkat7':
        _, _, test_loader = get_jurkat_loaders(batch_size, num_classes=7)
    elif dataset_name == 'sysmex':
        _, test_loader = get_sysmex_loaders(batch_size)
    elif dataset_name == 'sysmex4':
        _, _, test_loader = get_sysmex_7class_loaders(batch_size, num_classes=4)
    elif dataset_name == 'sysmex7':
        _, _, test_loader = get_sysmex_7class_loaders(batch_size, num_classes=7)
    elif dataset_name == 'phenocam':
        _, _, test_loader = get_phenocam_loaders(batch_size, label_type='season')
    elif dataset_name == 'phenocam_monthly':
        _, _, test_loader = get_phenocam_loaders(batch_size, label_type='month')
    return test_loader


def build_model(dataset_name, loss_key, device):
    cfg = DATASETS[dataset_name]
    if loss_key == 'vmce':
        model = VonMisesModel(cfg['channels'], cfg['num_classes'], cfg['size'])
    else:
        model = SimpleCNN(cfg['channels'], cfg['num_classes'], cfg['size'])
    return model.to(device)


def build_loss_fn(loss_key, num_classes, device):
    _, loss_fn_class = LOSS_FUNCTIONS[loss_key]
    if loss_key in VECTOR_LOSSES:
        return loss_fn_class(num_classes=num_classes).to(device)
    return loss_fn_class()


def plot_soft_confusion_matrix(soft_cm, class_names, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(soft_cm, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names,
                ax=ax)
    ax.set_xlabel('Predicted (Softmax)')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), required=True)
    parser.add_argument('--loss',    type=str, choices=LOSS_FUNCTIONS.keys(), required=True)
    parser.add_argument('--model_path', type=str, required=True,
                        help='保存済み.pthファイルのパス')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./eval_results')
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    cfg = DATASETS[args.dataset]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # データ・モデル・損失関数の準備
    print(f"Loading dataset: {args.dataset}")
    test_loader = get_test_loader(args.dataset, args.batch_size, args.seed)

    model = build_model(args.dataset, args.loss, device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Loaded weights: {args.model_path}")

    loss_fn = build_loss_fn(args.loss, cfg['num_classes'], device)

    # 評価
    print("Running evaluation...")
    metrics = evaluate_detailed(
        model, test_loader, loss_fn, device,
        cfg['num_classes'], cfg['class_names'], args.loss
    )

    # 結果の表示
    print(f"\nAccuracy:     {metrics['accuracy']:.4f}")
    print(f"Circular MAE: {metrics['circular_mae']:.4f}")
    print(f"\nSoft Confusion Matrix:")
    print(np.round(metrics['soft_confusion_matrix'], 3))

    # ソフト混同行列の保存
    model_stem = Path(args.model_path).stem  # ファイル名（拡張子なし）
    plot_soft_confusion_matrix(
        soft_cm=metrics['soft_confusion_matrix'],
        class_names=cfg['class_names'],
        title=f'Soft Confusion Matrix\n{args.dataset} / {args.loss} / {model_stem}',
        output_path=output_dir / f'soft_cm_{model_stem}.png'
    )

    # numpy形式でも保存（複数シードの比較などに使える）
    npy_path = output_dir / f'soft_cm_{model_stem}.npy'
    np.save(npy_path, metrics['soft_confusion_matrix'])
    print(f"Saved: {npy_path}")


if __name__ == '__main__':
    main()
