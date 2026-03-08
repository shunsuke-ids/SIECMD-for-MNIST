"""
VonMisesLearnedModel の学習済みμ配置の可視化スクリプト

学習によってクラスの配置角度μがどう変化したかを以下の観点で可視化する。
  1. 単位円上のμ配置: 学習済みμ（星）vs 等間隔（vmce）の理想位置（白丸）
     z分布のサンプル点も重ねて表示
  2. クラス間ギャップの棒グラフ: 学習後の各クラス間の弧長 vs 等間隔 (2π/C)

出力:
  - mu_unit_circle_{model_stem}.png : 単位円上のμ配置比較
  - mu_gaps_{model_stem}.png        : クラス間ギャップの棒グラフ
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
from pathlib import Path

from models import VonMisesLearnedModel
from datasets import get_jurkat_loaders, get_sysmex_7class_loaders, get_phenocam_loaders

DATASETS = {
    'jurkat4':          {'num_classes': 4,  'channels': 1, 'size': 66,  'class_names': ['G1', 'S', 'G2', 'M']},
    'jurkat7':          {'num_classes': 7,  'channels': 1, 'size': 66,  'class_names': ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']},
    'sysmex4':          {'num_classes': 4,  'channels': 3, 'size': 64,  'class_names': ['G1', 'S', 'G2', 'M']},
    'sysmex7':          {'num_classes': 7,  'channels': 3, 'size': 64,  'class_names': ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']},
    'phenocam':         {'num_classes': 4,  'channels': 3, 'size': 224, 'class_names': ['Spring', 'Summer', 'Fall', 'Winter']},
    'phenocam_monthly': {'num_classes': 12, 'channels': 3, 'size': 224, 'class_names': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']},
}


def get_test_loader(dataset_name, batch_size):
    if dataset_name == 'jurkat4':
        _, _, loader = get_jurkat_loaders(batch_size, num_classes=4)
    elif dataset_name == 'jurkat7':
        _, _, loader = get_jurkat_loaders(batch_size, num_classes=7)
    elif dataset_name == 'sysmex4':
        _, _, loader = get_sysmex_7class_loaders(batch_size, num_classes=4)
    elif dataset_name == 'sysmex7':
        _, _, loader = get_sysmex_7class_loaders(batch_size, num_classes=7)
    elif dataset_name == 'phenocam':
        _, _, loader = get_phenocam_loaders(batch_size)
    elif dataset_name == 'phenocam_monthly':
        _, _, loader = get_phenocam_loaders(batch_size, label_type='month')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return loader


def collect_z_values(model, loader, device):
    """テストセット全サンプルの z とラベルを収集"""
    model.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            z = model.backbone(inputs)  # (batch, 1)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
    all_z = np.concatenate(all_z).squeeze(1)   # (N,)
    all_labels = np.concatenate(all_labels)
    return all_z, all_labels



def plot_mu_unit_circle(mu_learned, mu_uniform, z_values, labels,
                        num_classes, class_names, output_dir, model_stem, dataset_name='',
                        max_samples=300):
    """
    単位円上にμ配置を可視化する。
      - z サンプル点: クラス色の散布図（半透明）
      - 等間隔μ (vmce): 白抜き丸（radius=1.10）
      - 学習済みμ (vmce_mu): 塗り潰し星（radius=1.20）
    """
    cmap = plt.cm.hsv
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(8, 8))

    # 単位円
    circle = Circle((0, 0), 1, fill=False, color='gray', linewidth=1.5, linestyle='--')
    ax.add_patch(circle)

    # z サンプル点（クラスごとに散布）
    for c in range(num_classes):
        mask = np.where(labels == c)[0]
        if len(mask) > max_samples:
            mask = np.random.choice(mask, max_samples, replace=False)
        z_c = z_values[mask]
        ax.scatter(np.cos(z_c), np.sin(z_c), c=[colors[c]], s=15, alpha=0.4, label=class_names[c])

    r_uniform = 1.10
    r_learned = 1.20
    r_label   = 1.38

    # 等間隔μ（白丸）
    for c in range(num_classes):
        xu, yu = np.cos(mu_uniform[c]) * r_uniform, np.sin(mu_uniform[c]) * r_uniform
        ax.scatter(xu, yu, s=120, facecolors='white', edgecolors=colors[c],
                   linewidths=2, zorder=9)

    # 学習済みμ（塗り潰し星）＋ 等間隔→学習済みへの矢印
    for c in range(num_classes):
        xu, yu = np.cos(mu_uniform[c]) * r_uniform, np.sin(mu_uniform[c]) * r_uniform
        xl, yl = np.cos(mu_learned[c]) * r_learned, np.sin(mu_learned[c]) * r_learned
        # 偏差矢印（等間隔 → 学習済み）
        ax.annotate('', xy=(xl, yl), xytext=(xu, yu),
                    arrowprops=dict(arrowstyle='->', color=colors[c], lw=1.2))
        # 星マーク
        ax.scatter(xl, yl, c=[colors[c]], s=250, marker='*',
                   edgecolors='black', linewidths=1, zorder=10)
        # クラス名ラベル
        ax.text(np.cos(mu_learned[c]) * r_label, np.sin(mu_learned[c]) * r_label,
                class_names[c], ha='center', va='center', fontsize=11, fontweight='bold')

    # 凡例のカスタムハンドル
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markeredgecolor='gray', markersize=10, label='uniform μ (vmce)'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray',
               markeredgecolor='black', markersize=14, label='learned μ (vmce_mu)'),
    ]

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axhline(0, color='lightgray', linewidth=0.5)
    ax.axvline(0, color='lightgray', linewidth=0.5)
    ax.set_title(f'[{dataset_name}] Learned μ on unit circle\n'
                 f'○: uniform (vmce), ★: learned (vmce_mu)', fontsize=13)
    ax.legend(handles=legend_handles + [ax.get_legend_handles_labels()[0][i]
              for i in range(num_classes)],
              fontsize=10, loc='upper right')

    plt.tight_layout()
    path = output_dir / f'mu_unit_circle_{model_stem}.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='VonMisesLearnedModel の学習済みμ配置の可視化')
    parser.add_argument('--model_path', type=str, required=True,
                        help='学習済みモデルの .pth ファイルパス (vmce_mu)')
    parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./mu_plots')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = DATASETS[args.dataset]
    num_classes = cfg['num_classes']
    class_names = cfg['class_names']

    # 等間隔μ（vmce の基準）
    mu_uniform = np.arange(num_classes) * (2.0 * np.pi / num_classes)

    # モデルのロード
    model_stem = Path(args.model_path).stem
    model = VonMisesLearnedModel(cfg['channels'], num_classes, cfg['size']).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {args.model_path}")

    # 学習済みμを取得
    mu_learned = model.von_mises_head.get_mu().detach().cpu().numpy()
    print(f"\n学習済みμ: {np.round(mu_learned, 4)}")
    print(f"等間隔μ  : {np.round(mu_uniform, 4)}")

    # データローダー（z分布の可視化用）
    test_loader = get_test_loader(args.dataset, args.batch_size)
    print(f"\nTest samples: {len(test_loader.dataset)}")
    z_values, labels = collect_z_values(model, test_loader, device)

    # 可視化
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_mu_unit_circle(mu_learned, mu_uniform, z_values, labels,
                        num_classes, class_names, output_dir, model_stem, dataset_name=args.dataset)

    print("\nDone!")


if __name__ == '__main__':
    main()
