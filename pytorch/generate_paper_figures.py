"""
論文用の混同行列図を生成するスクリプト

使い方:
    # Wandbから取得して生成
    python generate_paper_figures.py --project ce_vs_svl --output_dir ./paper_figures

    # 特定のrunのみ
    python generate_paper_figures.py --project ce_vs_svl --run_id abc123
"""

import argparse
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# データセットごとのクラス名
CLASS_NAMES = {
    'jurkat4': ['G1', 'S', 'G2', 'M'],
    'jurkat7': ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo'],
    'sysmex4': ['G1', 'S', 'G2', 'M'],
    'sysmex7': ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo'],
    'phenocam_monthly': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
}

# 対象のデータセットと損失関数
TARGET_DATASETS = ['sysmex7', 'jurkat7', 'sysmex4', 'jurkat4', 'phenocam_monthly']
TARGET_LOSSES = ['ce', 'msevl']


def plot_confusion_matrix_paper(cm, class_names, title, output_path, figsize=None):
    """論文用の混同行列を生成"""
    n_classes = len(class_names)

    # クラス数に応じてサイズ調整
    if figsize is None:
        if n_classes <= 4:
            figsize = (6, 5)
        elif n_classes <= 7:
            figsize = (8, 7)
        else:
            figsize = (10, 9)

    # フォントサイズ設定
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
    })

    fig, ax = plt.subplots(figsize=figsize)

    # アノテーションのフォントサイズ
    annot_fontsize = 14 if n_classes <= 7 else 10

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        annot_kws={'size': annot_fontsize},
        cbar_kws={'shrink': 0.8}
    )

    ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
    ax.set_ylabel('True', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)

    # ラベルの回転
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    # PDF保存
    fig.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    print(f"Saved: {output_path}")

    plt.close(fig)


def get_runs_from_wandb(project, entity=None):
    """Wandbからrunを取得"""
    api = wandb.Api()

    if entity:
        runs = api.runs(f"{entity}/{project}")
    else:
        runs = api.runs(project)

    return runs


def filter_runs(runs, target_datasets, target_losses):
    """対象のrunをフィルタリング"""
    filtered = []

    for run in runs:
        config = run.config
        dataset = config.get('dataset', '')
        loss = config.get('loss', '')

        if dataset in target_datasets and loss in target_losses:
            filtered.append(run)

    return filtered


def generate_figures_from_wandb(project, output_dir, entity=None, run_id=None):
    """Wandbから混同行列を取得して図を生成"""
    api = wandb.Api()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if run_id:
        # 特定のrunのみ
        if entity:
            run = api.run(f"{entity}/{project}/{run_id}")
        else:
            run = api.run(f"{project}/{run_id}")
        runs = [run]
    else:
        # 全run取得してフィルタリング
        runs = get_runs_from_wandb(project, entity)
        runs = filter_runs(runs, TARGET_DATASETS, TARGET_LOSSES)

    print(f"Found {len(runs)} matching runs")

    for run in runs:
        config = run.config
        dataset = config.get('dataset', '')
        loss = config.get('loss', '')

        # 混同行列データを取得
        summary = run.summary

        # best_confusion_matrixがあるか確認
        if 'best_confusion_matrix' not in summary:
            print(f"Skipping {run.name}: no confusion matrix data")
            continue

        cm = np.array(summary['best_confusion_matrix'])
        class_names = CLASS_NAMES.get(dataset, [str(i) for i in range(len(cm))])

        # 損失関数の表示名
        loss_display = {'ce': 'Cross-Entropy', 'msevl': 'MSE Vector Loss'}.get(loss, loss.upper())

        # タイトル
        title = f"{dataset.upper()} - {loss_display}"

        # ファイル名
        filename = f"confusion_matrix_{dataset}_{loss}.pdf"
        output_path = output_dir / filename

        plot_confusion_matrix_paper(cm, class_names, title, output_path)

    print(f"\nAll figures saved to: {output_dir}")


def generate_from_local_data(cm_data, dataset, loss, output_dir):
    """ローカルデータから図を生成（Wandbが使えない場合用）"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cm = np.array(cm_data)
    class_names = CLASS_NAMES.get(dataset, [str(i) for i in range(len(cm))])

    loss_display = {'ce': 'Cross-Entropy', 'msevl': 'MSE Vector Loss'}.get(loss, loss.upper())
    title = f"{dataset.upper()} - {loss_display}"

    filename = f"confusion_matrix_{dataset}_{loss}.pdf"
    output_path = output_dir / filename

    plot_confusion_matrix_paper(cm, class_names, title, output_path)


def main():
    parser = argparse.ArgumentParser(description='Generate paper figures from Wandb')
    parser.add_argument('--project', type=str, default='ce_vs_svl',
                        help='Wandb project name')
    parser.add_argument('--entity', type=str, default=None,
                        help='Wandb entity (username or team)')
    parser.add_argument('--output_dir', type=str, default='./paper_figures',
                        help='Output directory for figures')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Specific run ID to process')

    args = parser.parse_args()

    generate_figures_from_wandb(
        project=args.project,
        output_dir=args.output_dir,
        entity=args.entity,
        run_id=args.run_id
    )


if __name__ == '__main__':
    main()
