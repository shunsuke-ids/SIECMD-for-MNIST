#!/usr/bin/env python3
"""\
MNIST 通常分類 (10クラス) 用シンプルCNN スクリプト

既存の `mnist_fine_tuning.py` (円形回帰) とほぼ同じバックボーンを用いて、
出力層のみ 10 クラス softmax に変更した分類タスク版。

【本バージョン】ユーザー要望により画像出力 (サンプル可視化 / 混同行列) 機能を削除し、
精度指標 (accuracy) のみをシンプルに比較できる最小構成にしています。

機能:
- 学習 / 検証 / テスト (train:48k, val:12k, test:10k)
- 複数回実行で平均精度表示 (--runs)
- 最良 validation accuracy 重みの保存 (任意)

使用例:
    python src/regression/mnist_classification.py --epochs 10
    python src/regression/mnist_classification.py --epochs 5 --batch_size 64 --runs 3

"""
import os
import sys
import argparse
import numpy as np

from keras import models as km, layers as kl
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# プロジェクトルートをパスに追加（他モジュール利用時のための統一処理）
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# （既存 circular 回帰のユーティリティは本ファイルでは未使用だが、
#  もし後で切替実験するなら同様パス設定で統一性を保つ）

def create_simple_cnn_classifier(input_shape=(28, 28, 1), num_classes=10):
    """円形回帰と同様の軽量CNNで 10 クラス分類ヘッドを構築"""
    inputs = kl.Input(shape=input_shape)
    x = kl.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = kl.MaxPooling2D((2, 2))(x)
    x = kl.Conv2D(64, (3, 3), activation='relu')(x)
    x = kl.MaxPooling2D((2, 2))(x)
    x = kl.Conv2D(128, (3, 3), activation='relu')(x)
    x = kl.GlobalAveragePooling2D()(x)
    x = kl.Dense(256, activation='relu')(x)
    x = kl.Dense(128, activation='relu')(x)
    outputs = kl.Dense(num_classes, activation='softmax')(x)
    model = km.Model(inputs, outputs)
    return model


def load_and_prepare_mnist():
    """MNIST を読み込み、train/val/test に分割して返す"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # train/val 分割 (val 20%)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    return (x_train, x_val, x_test), (y_train, y_val, y_test)



def main():
    parser = argparse.ArgumentParser(description='MNIST 10クラス分類 (シンプルCNN)')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='学習エポック数')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='バッチサイズ')
    parser.add_argument('--runs', '-r', type=int, default=1, help='複数回実行 (平均化用)')
    # 画像出力系オプションは削除 (精度比較に特化)
    parser.add_argument('--save_weights', action='store_true', help='各 run の学習済み重みを保存')

    args = parser.parse_args()

    print('MNIST 分類学習開始')
    print(f'Epochs={args.epochs}, Batch={args.batch_size}, Runs={args.runs}')
    print('=' * 60)

    (x_train, x_val, x_test), (y_train, y_val, y_test) = load_and_prepare_mnist()
    print(f'Data shapes: Train={x_train.shape}, Val={x_val.shape}, Test={x_test.shape}')

    acc_results = np.zeros(args.runs, dtype=np.float32)

    for run in range(args.runs):
        print(f'Run {run + 1}/{args.runs}')

        model = create_simple_cnn_classifier()
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        callbacks_list = []
        weights_path = None
        if args.save_weights:
            os.makedirs('weights/mnist_cnn_cls', exist_ok=True)
            weights_path = f'weights/mnist_cnn_cls/mnist_class_run_{run}.keras'
            checkpoint = ModelCheckpoint(
                filepath=weights_path,
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            )
            callbacks_list.append(checkpoint)

        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        # ベスト重みロード (オプション)
        if args.save_weights and weights_path and os.path.exists(weights_path):
            print('Loading best weights for evaluation...')
            model.load_weights(weights_path)

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        acc_results[run] = test_acc
        print(f'Run {run + 1} Test Accuracy: {test_acc * 100:.2f}%')
        print('-' * 40)

        # 最終 run のみ簡易レポート (precision/recall/F1) を文字で表示
        if run == args.runs - 1:
            y_pred_probs = model.predict(x_test, verbose=0)
            y_pred = np.argmax(y_pred_probs, axis=1)
            print('Classification report:\n')
            print(classification_report(y_test, y_pred, digits=4))

    print('\n=== Summary ===')
    print(f'Accuracies: {[f"{a*100:.2f}%" for a in acc_results]}')
    print(f'Mean Accuracy: {acc_results.mean()*100:.2f}% ± {acc_results.std()*100:.2f}% (n={args.runs})')


if __name__ == '__main__':
    main()
