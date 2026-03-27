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
├── evaluate_saved.py           # 保存済みモデルからソフト混同行列を生成
├── visualize_unit_circle.py    # 単位円可視化
├── visualize_vonmises.py       # VonMisesModelのz分布可視化
├── visualize_mu.py             # VonMisesLearnedModelの学習済みμ配置の可視化
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
| `--lambda_circ` | float | - | 1.0 | `ce_msevl` の円形損失重み λ |
| `--kappa` | float | - | 1.0 | `vmsl` / `vmsl_k` の Von Mises 集中度パラメータ κ |

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
| `evaluate_detailed()` | 詳細評価。混同行列、ソフト混同行列、F1スコア、cMAE等を計算 |
| `get_vector_predictions()` | ベクトル損失用の予測関数。argmaxではなく距離最小で予測 |

### Early Stopping と Best Model 選択

2つの異なる基準を使い分けている。

| 判定 | 基準 | 条件 |
|------|------|------|
| Best Model 保存 | **val_acc** | 過去最高を更新したときにモデルを保存 |
| Early Stopping | **val_loss** | `patience` エポック連続で改善しなければ学習を打ち切り |

### オプティマイザ

| 項目 | 値 |
|------|-----|
| アルゴリズム | Adam |
| 学習率 | 0.001（`--lr` で変更可） |
| スケジューラ | なし（固定学習率） |

### ベクトル予測の仕組み

ベクトル損失（svl, nsvl, msevl, eucvl, arcvl）使用時:

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

### VonMisesModel

`SimpleCNN`（スカラー出力）と `VonMisesHead` を組み合わせたモデル。`--loss vmce` 指定時に使用。

```
入力画像
    │
    ├── SimpleCNN(num_classes=1)   # スカラー z を出力 (batch, 1)
    │
    └── VonMisesHead               # logit_c = κ · cos(z − μ_c) → (batch, num_classes)
```

κ（集中度パラメータ）は学習可能。出力は通常の CrossEntropyLoss と互換。

### VonMisesLearnedModel

`SimpleCNN`（スカラー出力）と `VonMisesLearnedHead` を組み合わせたモデル。`--loss vmce_mu` 指定時に使用。

```
入力画像
    │
    ├── SimpleCNN(num_classes=1)       # スカラー z を出力 (batch, 1)
    │
    └── VonMisesLearnedHead            # logit_c = cos(z − μ_c) → (batch, num_classes)
                                        # μ_c は学習可能（周期的順序を保持）
```

`VonMisesModel` との違い: κを固定（1.0）し、代わりにクラス配置角度μを学習可能パラメータにする。

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
| `arcvl` | `ArcDistanceVectorLoss` | 円弧距離（ラジアン）ベースのベクトル損失 |
| `vmce` | `VonMisesHead` + CE | Von Mises分布ロジット + CrossEntropyLoss（VonMisesModel使用） |
| `vmce_mu` | `VonMisesLearnedHead` + CE | μ学習可能版（VonMisesLearnedModel使用） |
| `slce` | `CircularSoftLabelCrossEntropyLoss` | 固定ソフトラベル（正解0.8・隣接各0.1）CE |
| `ce_msevl` | `CombinedCEMSEVectorLoss` | CE + λ×MSEVectorLoss の線形結合（`--lambda_circ` で λ 指定） |
| `ecdl` | `ExpectedCircularDistanceLoss` | Softmax確率で重み付けした期待循環距離を最小化 |
| `vmsl` | `VonMisesSoftLabelCELoss` | Von Mises分布ソフトラベル CE（κ固定、`--kappa` で指定、デフォルト1.0） |
| `vmsl_k` | `VonMisesSoftLabelCELoss` | Von Mises分布ソフトラベル CE（κを初期値から学習） |

### VonMisesSoftLabelCELoss の詳細

正解クラスを中心とした Von Mises 分布でソフトラベルを生成し、CE 損失を計算する。

```python
target_c = softmax(κ · cos(2π(c − y) / C))
```

κ が大きいほど one-hot に近づき、κ→0 で均一ラベルになる。

**8クラス時の確率分布（参考）**

| クラス距離 | κ=1 | κ=3 | κ=5 |
|-----------|-----|-----|-----|
| 0（正解） | 26.8% | 51.4% | 67.8% |
| ±1 | 20.0% | 21.4% | 15.7% |
| ±2 | 9.9% | 2.6% | 0.5% |
| ±3 | 4.9% | 0.3% | ~0% |
| 4（対角） | 3.6% | 0.1% | ~0% |

`vmsl_k` では κ を `log_κ` として保持し、学習中に更新する（正値保証）。wandb に毎エポック `kappa` としてログされる。

### VonMisesLearnedHead の詳細

μをパラメータとして学習する Von Mises ヘッド。周期的なクラス順序を破壊しないために、クラス間角度ギャップを softplus で正値化した累積和で表現する。

```python
# 学習パラメータ: raw_delta (C,)  初期値 0
delta = softplus(raw_delta)          # 正値化（全ギャップ等値スタート）
total = sum(delta)
mu[0] = 0                            # 回転の自由度を除去
mu[c] = cumsum(delta)[c-1] / total * 2π   # 累積比率で等分割

# ロジット計算（κ固定=1）
logits_c = cos(z - mu_c)
```

| 項目 | 内容 |
|------|------|
| 学習パラメータ | `raw_delta` (C 個のギャップ) |
| κ | 固定 1.0（`VonMisesHead` との違い） |
| μ[0] | 0 固定（回転の自由度を排除） |
| 初期配置 | 等間隔（`vmce` と同一スタート） |

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
| EucVL | `loss = mean(sqrt(sum((pred_vector - true_vector)²) + ε))` |
| ArcVL | `loss = mean(acos(clamp(dot(normalize(pred_vector), true_vector))))` |

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
| `cfv8` | 8 | 3 | 224 | Front, FR, Right, BR, Back, BL, Left, FL | `get_cfv_loader(num_classes=8)` |

### データ分割

| データセット | 分割方法 | 比率 | Stratify | Val有無 |
|-------------|---------|------|----------|---------|
| mnist | 組込み（torchvision） | 60,000 / 10,000 | - | なし（testで代用） |
| sysmex (3cls) | ディレクトリ分割済み | train/ / test/ | - | なし（testで代用） |
| jurkat, jurkat4, jurkat7 | sklearn 2段階分割 | 70% / 15% / 15% | あり | あり |
| sysmex4, sysmex7 | sklearn 2段階分割 | 70% / 15% / 15% | あり | あり |
| phenocam, phenocam_monthly | sklearn 2段階分割 | 70% / 15% / 15% | あり | あり |
| cfv8 | HuggingFace組込み分割 + val分割 | 80% / 20% / test | あり | あり |

> val_loader がないデータセット（mnist, sysmex 3cls）では、`val_loader = test_loader` として代用される（`train.py` L421-424）。

#### 70/15/15 分割の実装詳細（jurkat4/7, sysmex4/7, phenocam_monthly 共通）

5つのデータローダはすべて同一の2段階分割を使用している。

```python
# 第1段階: 全データ → train(70%) + temp(30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# 第2段階: temp(30%) → val(15%) + test(15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)
```

| 項目 | 値 |
|------|-----|
| random_state | 42（固定） |
| stratify | あり（クラスラベル `y` で層化抽出） |
| 分割タイミング | ラベルマージ（3/4クラス化）の**後** |

- **stratify=y**: 各クラスの比率が train/val/test で均等に保たれる
- ラベルマージ後に分割するため、マージ後のクラス比率に基づいて層化される

#### 各データローダの呼び出し対応

| train.py での指定 | 呼び出される関数 | 引数 |
|-------------------|-----------------|------|
| `--dataset jurkat4` | `get_jurkat_loaders()` | `num_classes=4` |
| `--dataset jurkat7` | `get_jurkat_loaders()` | `num_classes=7` |
| `--dataset sysmex4` | `get_sysmex_7class_loaders()` | `num_classes=4` |
| `--dataset sysmex7` | `get_sysmex_7class_loaders()` | `num_classes=7` |
| `--dataset phenocam_monthly` | `get_phenocam_loaders()` | `label_type='month'` |

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

## 5. visualize_mu.py - μ配置の可視化

`vmce_mu` で学習した `VonMisesLearnedModel` の学習済みμ配置を可視化する。

### 実行方法

```bash
python pytorch/visualize_mu.py --model_path <path_to_pth> --dataset <dataset>
```

### コマンドライン引数

| 引数 | 型 | 必須 | デフォルト | 説明 |
|------|-----|------|-----------|------|
| `--model_path` | str | ○ | - | `vmce_mu` で学習した `.pth` ファイルパス |
| `--dataset` | str | ○ | - | データセット名（jurkat4/7, sysmex4/7, phenocam, phenocam_monthly） |
| `--batch_size` | int | - | 64 | バッチサイズ |
| `--output_dir` | str | - | `./mu_plots` | 出力先ディレクトリ |

### 出力

| ファイル名 | 内容 |
|-----------|------|
| `mu_unit_circle_{model_stem}.png` | 単位円上のμ配置比較（等間隔○ vs 学習済み★、z分布散布図付き） |

---

## 6. metrics.py - 評価指標

### soft_confusion_matrix

クラスごとの平均Softmax分布（ソフト混同行列）を計算する。

```
soft_cm[i, j] = 真値クラスiのサンプルに対するクラスjへの平均Softmax確率
```

通常の混同行列（argmax予測）とは異なり、モデルが「どこに確率質量を置いているか」を示す。

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

## 7. 実行例

### 基本的な実行

```bash
# Cross Entropy で phenocam を学習
python pytorch/train.py --dataset phenocam --loss ce --epochs 50

# MSE Vector Loss で jurkat7 を学習
python pytorch/train.py --dataset jurkat7 --loss msevl --epochs 100 --lr 0.0005

# sysmex7 を早期終了付きで学習
python pytorch/train.py --dataset sysmex7 --loss ce --epochs 200 --patience 15

# vmce_mu（μ学習）で jurkat4 を学習
python pytorch/train.py --dataset jurkat4 --loss vmce_mu --epochs 100

# CFV データセット（8方向分類）を学習
python pytorch/train.py --dataset cfv8 --loss ce --epochs 50

# vmce_mu 学習済みモデルのμ配置を可視化
python pytorch/visualize_mu.py --model_path saved_models/jurkat4_vmce_mu_100ep_best.pth --dataset jurkat4
```

### 複数シード実行（スクリプト使用）

```bash
./run_experiments.sh phenocam ce 3  # 3回実行
```

---

## 8. 出力

### wandb ログ

**毎エポック**
- `train_loss`, `train_acc`: 学習損失・精度
- `val_loss`, `val_acc`: 検証損失・精度

**学習終了時（summary）**
- `best_test_acc`, `final_test_acc`: テスト精度
- `best_f1_macro`, `final_f1_macro`: マクロ平均F1
- `best_f1_weighted`, `final_f1_weighted`: 加重平均F1
- `best_circular_mae`, `final_circular_mae`: テストcMAE（サンプル平均）
- `best_circular_mae_macro`, `final_circular_mae_macro`: テストcMAE（マクロ平均）
- `confusion_matrix_best`, `confusion_matrix_final`: 混同行列画像
- `soft_confusion_matrix_best`, `soft_confusion_matrix_final`: ソフト混同行列画像

### 保存ファイル

```
saved_models/
├── {dataset}_{loss}_{epochs}ep_best.pth   # ベストモデル
└── {dataset}_{loss}_{epochs}ep_final.pth  # 最終モデル
```
