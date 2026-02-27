"""
VonMisesModelの潜在変数z分布の可視化スクリプト

学習済みモデルのバックボーンが出力するスカラーzが
各クラスのμ_c = 2π * c / C に対応した値に集まっているかを可視化する。

出力:
  - z_histogram.png   : クラスごとのz分布ヒストグラム（μ_c位置を縦線で表示）
  - z_unit_circle.png : zを単位円上の点として可視化（μ_c位置を星マークで表示）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
from pathlib import Path

from models import VonMisesModel
from datasets import get_jurkat_loaders, get_sysmex_loaders, get_sysmex_7class_loaders, get_phenocam_loaders

DATASETS = {
    'jurkat':           {'num_classes': 3,  'channels': 1, 'size': 66,  'class_names': ['G1', 'S', 'G2/M']},
    'jurkat4':          {'num_classes': 4,  'channels': 1, 'size': 66,  'class_names': ['G1', 'S', 'G2', 'M']},
    'jurkat7':          {'num_classes': 7,  'channels': 1, 'size': 66,  'class_names': ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']},
    'sysmex':           {'num_classes': 3,  'channels': 3, 'size': 64,  'class_names': ['G1', 'S', 'G2']},
    'sysmex4':          {'num_classes': 4,  'channels': 3, 'size': 64,  'class_names': ['G1', 'S', 'G2', 'M']},
    'sysmex7':          {'num_classes': 7,  'channels': 3, 'size': 64,  'class_names': ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']},
    'phenocam':         {'num_classes': 4,  'channels': 3, 'size': 224, 'class_names': ['Spring', 'Summer', 'Fall', 'Winter']},
    'phenocam_monthly': {'num_classes': 12, 'channels': 3, 'size': 224, 'class_names': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']},
}


def get_test_loader(dataset_name, batch_size):
    if dataset_name == 'jurkat':
        _, _, loader = get_jurkat_loaders(batch_size, num_classes=3)
    elif dataset_name == 'jurkat4':
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
    """テストセット全サンプルのzとラベルを収集"""
    model.eval()  # Dropoutをオフ、BatchNormを推論モードに（zが毎回変わるのを防ぐ）
    all_z, all_labels = [], []

    with torch.no_grad():  # 可視化では逆伝播不要なので計算グラフを作らない（メモリ節約）
        for inputs, labels in loader:
            inputs = inputs.to(device)
            z = model.backbone(inputs)  # VonMisesHeadを通さずバックボーンだけ呼ぶ → 生のスカラーz (batch, 1)
            all_z.append(z.cpu().numpy())  # GPUのtensorはそのままnumpyに変換できないので先にCPUに移す
            all_labels.append(labels.numpy())

    all_z = np.concatenate(all_z).squeeze(1)  # バッチごとのリストを結合(N,1)→squeeze→(N,)
    all_labels = np.concatenate(all_labels)
    return all_z, all_labels


def print_statistics(z_values, labels, num_classes, class_names, mu_c, kappa):
    """クラスごとのz統計とμ_cとの対応を表示"""
    print(f"\nκ = {kappa:.4f}")

    # 生のz値の全体統計
    print(f"\n【生のz値の範囲（mod 2π なし）】")
    print(f"  全体: min={z_values.min():.3f}, max={z_values.max():.3f}, "
          f"mean={z_values.mean():.3f}, std={z_values.std():.3f}")
    print(f"  {'クラス':>10} | {'min':>8} | {'max':>8} | {'mean':>8} | {'std':>8}")
    print("  " + "-" * 52)
    for c in range(num_classes):
        mask = labels == c # クラスcのサンプルだけTrueになるブール配列（例: labels=[0,1,0,2], c=0 → mask=[True, False, True, False]）
        z_c = z_values[mask] # ブール配列でzを絞り込み、クラスcのzだけ抽出（例: z_values=[0.1, 6.2, 0.3, 12.5], mask=[True, False, True, False] → z_c=[0.1, 0.3]）
        print(f"  {class_names[c]:>10} | {z_c.min():>8.3f} | {z_c.max():>8.3f} | "
              f"{z_c.mean():>8.3f} | {z_c.std():>8.3f}")

    print(f"\nクラスごとの z 統計 (z は mod 2π で表示)")
    print(f"{'クラス':>10} | {'μ_c':>8} | {'z 平均':>8} | {'z 中央値':>10} | {'μ_cとの差':>10}")
    print("-" * 60)

    z_wrapped = z_values % (2 * np.pi)

    for c in range(num_classes):
        mask = labels == c          # クラスcのサンプルだけTrueになるブール配列
        z_c = z_wrapped[mask]       # ブール配列でzを絞り込み、クラスcのzだけ抽出
        z_mean = z_c.mean()
        z_median = np.median(z_c)
        # 円環上の最短距離を計算（反対側を回るルートも考慮）
        # 例: z_mean=0.1, μ_c=6.1 のとき 普通の差=6.0 だが 2π-6.0=0.28 の方が近い
        diff = abs(z_mean - mu_c[c])
        diff = min(diff, 2 * np.pi - diff)
        print(f"{class_names[c]:>10} | {mu_c[c]:>8.3f} | {z_mean:>8.3f} | {z_median:>10.3f} | {diff:>10.3f}")


def plot_z_raw_histogram(z_values, labels, num_classes, class_names, kappa, output_dir, model_stem):
    """生のz値（mod 2π なし）のヒストグラム"""
    # HSVカラーマップを取得（色相環を一周する配色で、円環上に並ぶクラスの色分けに適している）
    cmap = plt.cm.hsv
    # cmap(0.0)〜cmap(1.0)の範囲でクラス数に合わせて均等に色を取り出す
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    # 描画領域を作る。fig はウィンドウ全体、ax はその中のグラフ本体
    fig, ax = plt.subplots(figsize=(10, 5))

    # 全クラス共通の bin 境界を事前に計算（これで棒の幅が統一される）
    bin_edges = np.linspace(z_values.min(), z_values.max(), 61)  # 60本の棒 → 境界点は61個

    for c in range(num_classes):
        mask = labels == c
        ax.hist(z_values[mask], bins=bin_edges,  # 共通の境界を渡すことで全クラスの棒幅を統一
                alpha=0.5,        # 透明度50%：複数クラスが重なっても後ろが透けて見えるように
                color=colors[c], # このクラスに対応する色
                label=class_names[c],  # ax.legend()で凡例に表示される名前
                density=True)    # 縦軸を確率密度にする（Falseだと頻度になり、クラス間のサンプル数差で高さが比べられなくなる）

    ax.set_xlabel('z (raw, no mod)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    # タイトルにκの値も埋め込む（:.3f で小数3桁）
    ax.set_title(f'Raw z distribution per class  (κ={kappa:.3f})', fontsize=14)
    # 各クラスの色とlabelで指定した名前の対応表（凡例）を描画する
    ax.legend(fontsize=11)

    # タイトルや軸ラベルが図の端で切れないように余白を自動調整する
    plt.tight_layout()
    # / 演算子は pathlib.Path の機能でパス結合になる
    # model_stem はモデルファイル名から拡張子を除いた部分（例：jurkat_vmce_100ep_best）
    path = output_dir / f'z_raw_histogram_{model_stem}.png'
    # bbox_inches='tight'：凡例や軸ラベルが切れないように図全体をぴったり囲むサイズで保存
    plt.savefig(path, bbox_inches='tight')
    # 図をメモリから解放する（しないと複数の図を連続描画したときメモリリークする）
    plt.close()
    print(f"Saved: {path}")


def plot_z_histogram(z_values, labels, num_classes, class_names, mu_c, kappa, output_dir, model_stem):
    """クラスごとのz分布ヒストグラム（μ_cを縦線で表示）"""
    cmap = plt.cm.hsv
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(10, 5))

    # zは実数全域だがcos(z)=cos(z+2π)なので周期2πで同値
    # 可視化時だけ[0, 2π]に巻き戻す（学習・推論ではそのままでよい）
    z_wrapped = z_values % (2 * np.pi)

    # 全クラス共通の bin 境界を [0, 2π] で事前に計算
    bin_edges = np.linspace(0, 2 * np.pi, 21)  # 20本の棒 → 境界点は21個

    for c in range(num_classes):
        mask = labels == c
        ax.hist(z_wrapped[mask], bins=bin_edges,
                alpha=0.5,       # 半透明にしてクラス間の重なりを見えるように
                color=colors[c],
                label=class_names[c],
                density=True)    # 縦軸を確率密度に正規化（クラス間のサンプル数差を吸収）

    # μ_cの理想位置を縦破線で表示
    for c in range(num_classes):
        ax.axvline(mu_c[c], color=colors[c], linestyle='--', linewidth=1.5)

    ax.set_xlabel('z (mod 2π)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.set_title(f'z distribution per class  (κ={kappa:.3f})', fontsize=14)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax.legend(fontsize=11)

    plt.tight_layout()
    path = output_dir / f'z_histogram_{model_stem}.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def plot_z_unit_circle(z_values, labels, num_classes, class_names, mu_c, kappa, output_dir, model_stem, max_samples=300):
    """zを単位円上の点として可視化（μ_cを星マークで表示）"""
    cmap = plt.cm.hsv
    colors = [cmap(i / num_classes) for i in range(num_classes)]

    fig, ax = plt.subplots(figsize=(8, 8))

    # fill=Falseで中を塗らない破線の円をパッチ（図形オブジェクト）として追加
    circle = Circle((0, 0), 1, fill=False, color='gray', linewidth=1.5, linestyle='--')
    ax.add_patch(circle)

    # クラスごとにサンプルを単位円上にプロット（サンプル数を制限）
    for c in range(num_classes):
        mask = np.where(labels == c)[0]  # クラスcのサンプルのインデックスを取得
        if len(mask) > max_samples:
            mask = np.random.choice(mask, max_samples, replace=False)  # 多すぎる場合はランダムに間引く
        z_c = z_values[mask]
        # cos/sinが周期性を処理するのでz % 2πの正規化は不要
        ax.scatter(np.cos(z_c), np.sin(z_c), c=[colors[c]], s=15,
                   alpha=0.4, label=class_names[c])

    # μ_cの位置を星マークで表示（半径1.12の位置に配置してサンプル点と重ならないようにする）
    star_r = 1.12
    for c in range(num_classes):
        xm, ym = np.cos(mu_c[c]) * star_r, np.sin(mu_c[c]) * star_r
        ax.scatter(xm, ym, c=[colors[c]], s=250, marker='*',
                   edgecolors='black', linewidths=1, zorder=10)  # zorder=10で他の点より手前に描画
        ax.text(xm * (1.22 / star_r), ym * (1.22 / star_r), class_names[c],  # 1.22倍の位置にラベル（円の外側）
                ha='center', va='center', fontsize=11, fontweight='bold')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.axhline(0, color='lightgray', linewidth=0.5)
    ax.axvline(0, color='lightgray', linewidth=0.5)
    ax.set_title(f'z on unit circle  (κ={kappa:.3f})', fontsize=14)
    ax.legend(fontsize=11, loc='upper right')

    plt.tight_layout()
    path = output_dir / f'z_unit_circle_{model_stem}.png'
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description='VonMisesModelのz分布可視化')
    parser.add_argument('--model_path', type=str, required=True,
                        help='学習済みモデルの .pth ファイルパス')
    parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), required=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--output_dir', type=str, default='./vonmises_plots')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = DATASETS[args.dataset]
    num_classes = cfg['num_classes']
    class_names = cfg['class_names']

    # μ_c: クラスの理想角度
    mu_c = np.arange(num_classes) * (2.0 * np.pi / num_classes)

    # モデルのロード
    model_stem = Path(args.model_path).stem  # 例: "jurkat_vmce_100ep_best"
    model = VonMisesModel(cfg['channels'], num_classes, cfg['size']).to(device) # モデルの初期化（アーキテクチャはデータセットに合わせて）
    state_dict = torch.load(args.model_path, map_location=device) # 学習済みモデルの重みをロード
    model.load_state_dict(state_dict) # モデルに重みを適用
    print(f"Loaded model from {args.model_path}")

    kappa = model.von_mises_head.kappa.item() # 1要素のtensorをPythonのスカラー値に変換するメソッド

    # データローダー
    test_loader = get_test_loader(args.dataset, args.batch_size)
    print(f"Test samples: {len(test_loader.dataset)}")

    # z値の収集
    z_values, labels = collect_z_values(model, test_loader, device)

    # 統計の表示
    print_statistics(z_values, labels, num_classes, class_names, mu_c, kappa)

    # 可視化
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_z_raw_histogram(z_values, labels, num_classes, class_names, kappa, output_dir, model_stem)
    plot_z_histogram(z_values, labels, num_classes, class_names, mu_c, kappa, output_dir, model_stem)
    plot_z_unit_circle(z_values, labels, num_classes, class_names, mu_c, kappa, output_dir, model_stem)

    print("\nDone!")


if __name__ == '__main__':
    main()
