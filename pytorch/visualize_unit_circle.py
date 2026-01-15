"""
単位円上でのサンプルベクトル可視化スクリプト

学習の進行に伴い、サンプルの予測ベクトルが中心付近から
各真値クラスの位置へ近づいていく様子を可視化する
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
from pathlib import Path

from models import SimpleCNN
from datasets import get_phenocam_loaders
from losses import SoftmaxVectorLoss, NormalizedSoftmaxVectorLoss, MSEVectorLoss, EuclideanVectorLoss

LOSS_FUNCTIONS = {
    'ce': ('CrossEntropyLoss', nn.CrossEntropyLoss),
    'svl': ('SoftmaxVectorLoss', SoftmaxVectorLoss),
    'nsvl': ('NormalizedSoftmaxVectorLoss', NormalizedSoftmaxVectorLoss),
    'msevl': ('MSEVectorLoss', MSEVectorLoss),
    'eucvl': ('EuclideanVectorLoss', EuclideanVectorLoss)
}

MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def get_class_coords(num_classes):
    """クラスの単位円上の座標を取得"""
    angles = torch.arange(num_classes, dtype=torch.float32) * (2.0 * np.pi / num_classes)
    class_coords = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
    return class_coords


def compute_pred_vectors(model, loader, device, num_classes):
    """各サンプルの予測ベクトルと真値ラベルを計算"""
    model.eval()
    class_coords = get_class_coords(num_classes).to(device)

    all_pred_vectors = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            softmax_out = F.softmax(logits, dim=1)
            pred_vectors = torch.matmul(softmax_out, class_coords)

            all_pred_vectors.append(pred_vectors.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_pred_vectors), np.concatenate(all_labels)


def plot_unit_circle(pred_vectors, labels, num_classes, epoch, output_dir,
                     max_samples=200, loss_name=''):
    """単位円上にサンプルをプロット"""
    class_coords = get_class_coords(num_classes).numpy()

    # カラーマップ（12クラス用）
    cmap = plt.cm.hsv
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 10))

    # 単位円を描画
    circle = Circle((0, 0), 1, fill=False, color='gray', linewidth=2, linestyle='--')
    ax.add_patch(circle)

    # クラス位置をマーカーで表示
    for i, (x, y) in enumerate(class_coords):
        ax.scatter(x, y, c=[colors[i]], s=300, marker='s', edgecolors='black',
                   linewidths=2, zorder=10)
        # ラベルを少し外側に配置
        label_x = x * 1.15
        label_y = y * 1.15
        ax.text(label_x, label_y, MONTH_NAMES[i], ha='center', va='center',
                fontsize=12, fontweight='bold')

    # サンプル数を制限
    if len(pred_vectors) > max_samples:
        indices = np.random.choice(len(pred_vectors), max_samples, replace=False)
        pred_vectors = pred_vectors[indices]
        labels = labels[indices]

    # 各サンプルをプロット
    for i in range(len(pred_vectors)):
        x, y = pred_vectors[i]
        label = labels[i]
        ax.scatter(x, y, c=[colors[label]], s=50, alpha=0.6, edgecolors='none')

    # 原点にマーカー
    ax.scatter(0, 0, c='black', s=100, marker='+', linewidths=2, zorder=5)

    # 軸の設定
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='lightgray', linestyle='-', linewidth=0.5)
    ax.axvline(x=0, color='lightgray', linestyle='-', linewidth=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)

    title = f'Unit Circle - Epoch {epoch}'
    if loss_name:
        title += f' ({loss_name})'
    ax.set_title(title, fontsize=14)

    # 凡例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=colors[i], markersize=10,
                                   label=MONTH_NAMES[i])
                       for i in range(num_classes)]
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
              fontsize=10, title='True Class')

    plt.tight_layout()

    # 保存
    output_path = output_dir / f'unit_circle_epoch_{epoch:03d}.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def train_epoch(model, loader, loss_fn, optimizer, device):
    """1エポックの学習"""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='単位円上でのベクトル可視化')
    parser.add_argument('--loss', type=str, choices=LOSS_FUNCTIONS.keys(),
                        default='svl', help='損失関数')
    parser.add_argument('--epochs', type=int, default=30, help='総エポック数')
    parser.add_argument('--batch_size', type=int, default=64, help='バッチサイズ')
    parser.add_argument('--lr', type=float, default=0.001, help='学習率')
    parser.add_argument('--plot_epochs', type=int, nargs='+', default=[0, 5, 10, 20],
                        help='可視化するエポック（0は学習前）')
    parser.add_argument('--max_samples', type=int, default=200,
                        help='プロットする最大サンプル数')
    parser.add_argument('--output_dir', type=str, default='./unit_circle_plots',
                        help='出力ディレクトリ')
    parser.add_argument('--limit', type=int, default=None,
                        help='クラスごとのサンプル数制限')

    args = parser.parse_args()

    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # データセット読み込み（月ごと12クラス）
    num_classes = 12
    train_loader, val_loader, test_loader = get_phenocam_loaders(
        batch_size=args.batch_size,
        limit_per_season=args.limit,
        label_type='month'
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # モデル初期化
    model = SimpleCNN(input_channels=3, num_classes=num_classes, image_size=224).to(device)

    # 損失関数
    loss_name, loss_fn_class = LOSS_FUNCTIONS[args.loss]
    if args.loss in ['svl', 'nsvl', 'msevl', 'eucvl']:
        loss_fn = loss_fn_class(num_classes=num_classes).to(device)
    else:
        loss_fn = loss_fn_class()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nTraining with {loss_name} for {args.epochs} epochs")
    print(f"Plot epochs: {args.plot_epochs}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # 学習前（epoch 0）の可視化
    if 0 in args.plot_epochs:
        pred_vectors, labels = compute_pred_vectors(model, val_loader, device, num_classes)
        plot_unit_circle(pred_vectors, labels, num_classes, 0, output_dir,
                         args.max_samples, loss_name)

    # 学習ループ
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        print(f"Epoch {epoch:2d}/{args.epochs} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")

        # 指定エポックで可視化
        if epoch in args.plot_epochs:
            pred_vectors, labels = compute_pred_vectors(model, val_loader, device, num_classes)
            plot_unit_circle(pred_vectors, labels, num_classes, epoch, output_dir,
                             args.max_samples, loss_name)

    print("\nDone!")


if __name__ == '__main__':
    main()
