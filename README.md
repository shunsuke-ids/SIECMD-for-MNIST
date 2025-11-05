# SIECMD
This repository provides the implementation of the presented solution in: 
[Single Image Estimation of Cell Migration Direction by Deep Circular Regression](https://arxiv.org/abs/2406.19162) (L. Bruns et al.) [preprint manuscript].

## Abstract
In this paper we study the problem of estimating the migration direction of cells based on a single image. To the best of our knowledge, there is only one related work that uses a classification CNN for four classes (quadrants). This approach does not allow detailed directional resolution. We solve the single image estimation problem using deep circular regression with special attention to cycle-sensitive methods. On two databases we achieve an average accuracy of ∼17 degrees, which is a significant improvement over the previous work. 

## Implementation
Following, the implementation of SIECMD is described as directory structure. 

### /src/DL
This directory contains all deep learning based solution parts. 
The evaluation metrics are implemented in [metrics.py](./src/DL/metrics.py). 
The activation and loss functions that support circular regression are implemented in [activation_functions.py](./src/DL/activation_functions.py) and [losses.py](./src/DL/losses.py). The file [models.py](./src/DL/models.py) contains the probing model for the described parameter probing. The file also features a classification model of similar size. 

### /src/preprocessing
This directory features all methods for preprocessing and handling datasets before training. The file [augment.py](./src/preprocessing/augment.py) implements die augmentation process for training data preperation as well as test-time augmentation (TTA). The methods in [handle_dataset.py](./src/preprocessing/handle_dataset.py) can be used to prepare datasets to split into train, test and val set or normal distribute angular representations. 

### /src/regression
This directory contains implementations of SIECMD regression task. The files [fine_tuning.py](./src/regression/fine_tuning.py) and [probing.py](./src/regression/probing.py) are examples on how to train and evaluate the circular regression models. New datasets can be prepared to match the format used in those examples by using [prepare_dataset.py](./src/regression/prepare_dataset.py). The remaining two files contain helper functions, also used in the two example applications. The file [format_gt.py](./src/regression/format_gt.py) supports the conversion between different ground truth encodings. Finally, the file [circular_operations.py](./src/regression/circular_operations.py) contains methods for circular averaging which are used for TTA.

### /weights
This is the default weight-file save directory. Files are saved as *.keras* files and can be loaded to keras models (see example [fine_tuning.py](./src/regression/fine_tuning.py)). 

## プロジェクト早見表（日本語）: ファイル構成と役割

リポジトリの現状の構成と、主要ファイルの役割を日本語でまとめます。

トップレベル
- `requirements.txt` — 実行に必要なライブラリ（Keras/TensorFlow, scikit-learn, OpenCV など）
- `environment.yml` — Conda 環境の定義（任意）
- `setup_gpu_env.sh` — GPU 環境セットアップ用の補助スクリプト
- `weights/` — 学習済みチェックポイント（`.keras`）の保存先
- `figs/` — 図の保存先（git ignore 対象）

DL 基盤（`src/DL` と `src/preprocessing`）
- `src/DL/activation_functions.py` — 円周（角度）回りの出力に対応するカスタム活性化
- `src/DL/losses.py` — 円環回帰向け損失（例: 線形距離二乗）
- `src/DL/metrics.py` — 評価ユーティリティ（平均角度偏差など）
- `src/DL/models.py` — プロービング等で用いるモデル定義
- `src/preprocessing/augment.py` — 学習時・テスト時（TTA）の拡張
- `src/preprocessing/handle_dataset.py` — 前処理・分割などのヘルパ

回帰/分類ワークフロー（`src/regression`）
- `jurkat_cyclic_regression.py` — Jurkat（CellCycle）7 クラスの円環回帰
	- Ch3（明視野）画像を読み込み、7 フェーズを単位円に等間隔マッピング
	- 指標: 平均角度偏差、トレランス精度（±180/7°）
	- 予測角を最近傍のクラス中心にスナップして混同行列（CSV）出力も可能
- `jurkat_classification_baseline.py` — 同一バックボーンの 7 クラス分類ベースライン
	- 指標: 分類精度。fold ごとに混同行列（CSV）出力対応
- `plot_confusion_matrices.py` — 混同行列 CSV をヒートマップ PNG に可視化
	- `--root` 配下を探索し `confusion_matrix.csv` を見つけて図化、`--out_dir` に保存
- `aggregate_confusion_matrics.py` — fold ごとの混同行列を合算し 1 枚に集約
	- `<root>/fold*/confusion_matrix.csv` を読み込み、合計（raw）と正規化版を保存
	- 併せて `metrics.json`（total/correct/errors/accuracy, 入力ファイル一覧）を出力
	- オプション: `--normalize {none,true,pred,all}`
- `format_gt.py` — 角度↔点 変換などの幾何ヘルパ
- `fine_tuning.py`, `probing.py` — SIECMD の学習/評価例
- `prepare_dataset.py` — データ整形ユーティリティ
- `jurkat_visualize.py` — 重み/特徴の可視化補助
- `mnist_classification.py`, `mnist_fine_tuning.py` — MNIST 向けの参考スクリプト
- `results/`, `figs/`, `weights/` — 実行結果の保存先（git ignore）

出力（混同行列関連）
- 各 fold の混同行列（非正規化推奨）:
	- `src/regression/results/confmats_none/classification/fold*/confusion_matrix.csv`
	- `src/regression/results/confmats_none/regression/fold*/confusion_matrix.csv`
- 合算（`aggregate_confusion_matrics.py` の出力）:
	- `.../combined/confusion_matrix_raw.csv` — 各 fold の生カウント合計
	- `.../combined/confusion_matrix.csv` — 正規化モードに応じて規格化
	- `.../combined/metrics.json` — 件数・精度等の集約メタ情報
- PNG 化（`plot_confusion_matrices.py`）:
	- `src/regression/figs/confmats_none/{classification,regression}/fold*/confusion_matrix.png`
	- 合算版は対応する `.../combined/` 配下に出力

備考
- 生成物（results, figs, weights, cache 等）は `.gitignore` 済みです。
- README 内の一部は原著 SIECMD 論文の例に基づく記述があり、Jurkat 向けの 7 フェーズ実験用スクリプトは追加分です。
