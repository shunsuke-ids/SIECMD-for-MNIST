# PyTorch 実装ドキュメント

本ディレクトリは、周期的クラス分類のためのベクトル損失関数の実験コードを含む。

---

## ディレクトリ構成

```
pytorch/
├── train.py                    # メイン学習スクリプト
├── models.py                   # CNNモデル定義
├── losses.py                   # 損失関数定義
├── datasets.py                 # データセットローダー
├── metrics.py                  # 評価指標
├── visualize_unit_circle.py    # 単位円可視化
└── generate_paper_figures.py   # 論文用図表生成
```

---

## 1. train.py - メイン学習スクリプト

### 実行方法

```bash
python pytorch/train.py --dataset <dataset> --loss <loss> [options]
```

### コマンドライン引数

| 引数 | 型 | 必須 | デフォルト | 説明 |
|------|-----|------|-----------|------|
| `--dataset` | str | ○ | - | データセット名 |
| `--loss` | str | ○ | - | 損失関数名 |
| `--epochs` | int | - | 20 | エポック数 |
| `--batch_size` | int | - | 64 | バッチサイズ |
| `--lr` | float | - | 0.001 | 学習率 |
| `--patience` | int | - | 8 | Early Stopping patience |
| `--limit_per_phase` | int | - | None | クラスあたりのサンプル数上限 |
| `--seed` | int | - | 42 | 乱数シード |

### 処理フロー

```
main()
  │
  ├── 1. set_seed()                 # 再現性のためシード設定
  │
  ├── 2. データローダー取得          # データセットに応じた関数を呼び出し
  │
  ├── 3. wandb.init()               # 実験ログ開始
  │
  ├── 4. SimpleCNN初期化            # モデル作成
  │
  ├── 5. 損失関数・オプティマイザ設定
  │
  └── 6. train_and_evaluate()
          │
          ├── エポックループ
          │     ├── train_epoch()   # 1エポック学習
          │     ├── evaluate()      # 検証セット評価
          │     ├── ベストモデル保存（val_acc基準）
          │     └── Early Stopping判定（val_loss基準）
          │
          └── テスト評価
                ├── evaluate_detailed() (ベストモデル)
                └── evaluate_detailed() (最終モデル)
```

### 主要関数

| 関数 | 説明 |
|------|------|
| `train_epoch()` | 1エポックの学習を実行。損失と精度を返す |
| `evaluate()` | 検証/テスト評価。ベクトル損失の場合は距離ベース予測を使用 |
| `evaluate_detailed()` | 詳細評価。混同行列、F1スコア、cMAE等を計算 |
| `get_vector_predictions()` | ベクトル損失用の予測関数。argmaxではなく距離最小で予測 |

### ベクトル予測の仕組み

ベクトル損失（svl, nsvl, msevl, eucvl）使用時:

1. 各クラスを単位円上に等間隔で配置
2. softmax確率で重み付けした予測ベクトルを計算
3. 各クラスベクトルとの距離で最終予測を決定

```python
# クラス座標の計算
angles = [0, 2π/n, 4π/n, ..., 2π(n-1)/n]
class_coords = [(cos(θ), sin(θ)) for θ in angles]

# 予測ベクトル
pred_vector = Σ(softmax[i] * class_coords[i])

# 最終予測 = 最も近いクラス
prediction = argmin(distance(pred_vector, class_coords))
```

---

## 2. models.py - モデル定義

### SimpleCNN

シンプルな畳み込みニューラルネットワーク。

```
入力画像
    │
    ├── Conv2d(in, 16, kernel=5) + ReLU
    ├── MaxPool2d(2, 2)
    │
    ├── Conv2d(16, 32, kernel=5) + ReLU
    ├── MaxPool2d(2, 2)
    │
    ├── Flatten
    │
    ├── Linear(flatten_size, 256) + ReLU
    ├── Dropout(0.5)
    ├── Linear(256, 16) + ReLU
    └── Linear(16, num_classes)
```

### コンストラクタ引数

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `input_channels` | 1 | 入力チャンネル数 |
| `num_classes` | 10 | 出力クラス数 |
| `image_size` | 28 | 入力画像サイズ |

---

## 3. losses.py - 損失関数

### 損失関数一覧

| キー | クラス名 | 説明 |
|------|---------|------|
| `ce` | `CrossEntropyLoss` | 標準的な分類損失（PyTorch組込み） |
| `svl` | `SoftmaxVectorLoss` | 内積ベースのベクトル損失 |
| `nsvl` | `NormalizedSoftmaxVectorLoss` | 正規化版SVL |
| `msevl` | `MSEVectorLoss` | MSEベースのベクトル損失 |
| `eucvl` | `EuclideanVectorLoss` | ユークリッド距離ベースのベクトル損失 |

### ベクトル損失の共通構造

```python
# 1. クラス座標の初期化（コンストラクタで1回だけ）
angles = [0, 2π/n, 4π/n, ...]
class_coords = [(cos(θ), sin(θ)), ...]  # 単位円上に配置

# 2. 予測ベクトルの計算
pred_softmax = softmax(logits)
pred_vector = pred_softmax @ class_coords  # 重み付き合成

# 3. 真値ベクトルの計算
true_onehot = one_hot(labels)
true_vector = true_onehot @ class_coords  # 真のクラス座標

# 4. 損失計算（損失関数ごとに異なる）
```

### 各損失関数の計算式

| 損失関数 | 計算式 |
|---------|--------|
| SVL | `loss = 1 - dot(pred_vector, true_vector)` |
| NSVL | `loss = 1 - dot(normalize(pred_vector), true_vector)` |
| MSEVL | `loss = mean(sum((pred_vector - true_vector)²))` |
| EucVL | `loss = mean(sqrt(sum((pred_vector - true_vector)²)))` |

---

## 4. datasets.py - データセットローダー

### 対応データセット一覧

| データセット | クラス数 | チャンネル | サイズ | クラス名 | ローダー関数 |
|-------------|---------|-----------|--------|---------|-------------|
| `mnist` | 10 | 1 | 28 | 0-9 | `get_mnist_loaders()` |
| `jurkat` | 3 | 1 | 66 | G1, S, G2/M | `get_jurkat_loaders(num_classes=3)` |
| `jurkat4` | 4 | 1 | 66 | G1, S, G2, M | `get_jurkat_loaders(num_classes=4)` |
| `jurkat7` | 7 | 1 | 66 | G1, S, G2, Pro, Meta, Ana, Telo | `get_jurkat_loaders(num_classes=7)` |
| `sysmex` | 3 | 3 | 64 | G1, S, G2 | `get_sysmex_loaders()` |
| `sysmex4` | 4 | 3 | 64 | G1, S, G2, M | `get_sysmex_7class_loaders(num_classes=4)` |
| `sysmex7` | 7 | 3 | 64 | G1, S, G2, Pro, Meta, Ana, Telo | `get_sysmex_7class_loaders(num_classes=7)` |
| `phenocam` | 4 | 3 | 224 | Spring, Summer, Fall, Winter | `get_phenocam_loaders(label_type='season')` |
| `phenocam_monthly` | 12 | 3 | 224 | Jan-Dec | `get_phenocam_loaders(label_type='month')` |

### データ分割

| データセット | 分割方法 | 比率 |
|-------------|---------|------|
| mnist | 組込み | train / test |
| sysmex (3cls) | ディレクトリ分割済み | train / test |
| その他 | train_test_split | 70% / 15% / 15% (train/val/test) |

### ラベルマッピング

**4クラス化 (`merge_4class`)**
```
G1 → G1, S → S, G2 → G2, Pro/Meta/Ana/Telo → M
```

**3クラス化 (`merge_jurkat_3class`)** ※Jurkatのみ
```
G1 → G1, S → S, G2/Pro/Meta/Ana/Telo → G2/M
```

### Datasetクラス

| クラス | 用途 | 正規化 |
|--------|------|--------|
| `ImageDataset` | 一般画像 | 0-255 → 0-1 |
| `NormalizedImageDataset` | 正規化済み画像（phenocam） | なし |

---

## 5. metrics.py - 評価指標

### circular_mae

周期性を考慮したMAE（Mean Absolute Error）。

```python
diff = |y_pred - y_true|
circular_diff = min(diff, num_classes - diff)
cMAE = mean(circular_diff)
```

**例: 4クラス（Spring=0, Summer=1, Fall=2, Winter=3）**
- Winter(3) → Spring(0) の誤差: min(3, 4-3) = 1
- Summer(1) → Winter(3) の誤差: min(2, 4-2) = 2

### circular_mae_per_class

クラスごとのcMAEを計算し、マクロ平均も返す。

```python
{
    'per_class': {0: 0.5, 1: 0.3, ...},  # 各クラスのcMAE
    'macro': 0.4                          # マクロ平均
}
```

---

## 6. 実行例

### 基本的な実行

```bash
# Cross Entropy で phenocam を学習
python pytorch/train.py --dataset phenocam --loss ce --epochs 50

# MSE Vector Loss で jurkat7 を学習
python pytorch/train.py --dataset jurkat7 --loss msevl --epochs 100 --lr 0.0005

# sysmex7 を早期終了付きで学習
python pytorch/train.py --dataset sysmex7 --loss ce --epochs 200 --patience 15
```

### 複数シード実行（スクリプト使用）

```bash
./run_experiments.sh phenocam ce 3  # 3回実行
```

---

## 7. 出力

### wandb ログ

- `train_loss`, `train_acc`: 学習損失・精度（毎エポック）
- `val_loss`, `val_acc`: 検証損失・精度（毎エポック）
- `best_test_acc`, `final_test_acc`: テスト精度
- `best_circular_mae`, `final_circular_mae`: テストcMAE
- `confusion_matrix_best`, `confusion_matrix_final`: 混同行列画像

### 保存ファイル

```
saved_models/
├── {dataset}_{loss}_{epochs}ep_best.pth   # ベストモデル
└── {dataset}_{loss}_{epochs}ep_final.pth  # 最終モデル
```
