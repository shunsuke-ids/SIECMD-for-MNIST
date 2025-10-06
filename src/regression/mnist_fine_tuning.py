#!/usr/bin/env python3
"""
MNIST用円形回帰モデル（シンプルCNN版）
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
"""Reverted to version without explicit reproducibility / CSV logging features."""

from keras import models as km, layers as kl, callbacks
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

# プロジェクトルートをパスに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.DL.metrics import prediction_mean_deviation
from src.DL.activation_functions import sigmoid_activation
from src.DL.losses import linear_dist_squared_loss
from src.regression.format_gt import angles_2_unit_circle_points, points_2_angles, associated_points_on_circle


def create_simple_cnn(input_shape=(28, 28, 1)):
    """シンプルなCNNバックボーンを作成"""
    inputs = kl.Input(shape=input_shape)
    
    # CNN layers
    x = kl.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = kl.MaxPooling2D((2, 2))(x)
    x = kl.Conv2D(64, (3, 3), activation='relu')(x)
    x = kl.MaxPooling2D((2, 2))(x)
    x = kl.Conv2D(128, (3, 3), activation='relu')(x)
    x = kl.GlobalAveragePooling2D()(x)
    
    # Regression head for circular output
    x = kl.Dense(256, activation='relu')(x)
    x = kl.Dense(128, activation='relu')(x)
    outputs = kl.Dense(2, activation=sigmoid_activation)(x)
    
    model = km.Model(inputs, outputs)
    return model


def prepare_mnist_data(similarity_based=False):
    """MNISTデータを円形回帰用に準備"""
    # MNISTデータ読み込み
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # 正規化
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 次元追加 (28, 28) -> (28, 28, 1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    
    # 角度配置の選択
    if similarity_based:
        # 視覚的類似性に基づく角度配置
        angle_mapping = {
            0: 0,     # 0: 円形
            6: 36,    # 6: 円形の変形
            9: 72,    # 9: 6に類似
            8: 108,   # 8: 複雑な形状
            3: 144,   # 3: 8に若干類似
            2: 180,   # 2: 角的
            5: 216,   # 5: 6の反転系
            1: 252,   # 1: 線的
            7: 288,   # 7: 1に類似（角的線）
            4: 324    # 4: 角的
        }
        print("視覚的類似性に基づく角度配置を使用")
    else:
        # 従来の等間隔配置
        angle_mapping = {i: i * 36.0 for i in range(10)}
        print("従来の等間隔角度配置を使用")
    
    # ラベルを角度に変換
    angles_train = np.array([angle_mapping[label] for label in y_train])
    angles_test = np.array([angle_mapping[label] for label in y_test])
    
    # 角度を単位円上の点に変換
    y_train_circular = angles_2_unit_circle_points(angles_train)
    y_test_circular = angles_2_unit_circle_points(angles_test)
    
    # train/val分割
    x_train, x_val, y_train_circular, y_val_circular = train_test_split(
        x_train, y_train_circular, test_size=0.2, random_state=42
    )
    
    return (x_train, x_val, x_test), (y_train_circular, y_val_circular, y_test_circular), angles_test


## (Seed utility removed per user request)


def get_digit_from_angle(angle, similarity_based=False):
    """角度から対応する数字を取得"""
    if similarity_based:
        angle_to_digit = {
            0: 0, 36: 6, 72: 9, 108: 8, 144: 3,
            180: 2, 216: 5, 252: 1, 288: 7, 324: 4
        }
    else:
        angle_to_digit = {i * 36: i for i in range(10)}
    
    # 最も近い角度を見つける
    closest_angle = min(angle_to_digit.keys(), key=lambda x: min(abs(angle - x), abs(angle - x + 360), abs(angle - x - 360)))
    return angle_to_digit[closest_angle]


def angle_to_label_with_tolerance(predicted_angle, true_angle, tolerance=18.0):
    """
    予測角度が真の角度から許容範囲内にあるかチェック
    
    Args:
        predicted_angle: 予測された角度 (0-360)
        true_angle: 真の角度 (0-360)
        tolerance: 許容角度範囲 (デフォルト±18°)
    
    Returns:
        bool: 許容範囲内ならTrue
    """
    # 円形距離を計算（0°と360°の境界を考慮）
    diff = abs(predicted_angle - true_angle)
    circular_diff = min(diff, 360 - diff)
    
    return circular_diff <= tolerance


def calculate_angle_accuracy(predicted_angles, true_angles, tolerance=18.0):
    """
    角度予測の分類精度を計算（±tolerance度の許容範囲で）
    
    Args:
        predicted_angles: 予測角度配列
        true_angles: 正解角度配列  
        tolerance: 許容角度範囲 (デフォルト±18°)
    
    Returns:
        float: 精度 (0.0-1.0)
    """
    correct_count = 0
    total_count = len(predicted_angles)
    
    for pred_angle, true_angle in zip(predicted_angles, true_angles):
        if angle_to_label_with_tolerance(pred_angle, true_angle, tolerance):
            correct_count += 1
    
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    return accuracy


def visualize_digits_on_circle(x_test, y_test, predictions, angles_test, predicted_angles, num_samples=5, epoch=None):
    """数字ごとの画像を単位円上にプロットして視覚化"""
    plt.figure(figsize=(15, 15))
    
    # y_testをnumpy配列に変換（リストの場合）
    if isinstance(y_test, list):
        y_test = np.array(y_test)
    
    # 単位円を描画
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)
    
    # 角度のマーカーを描画（0°, 36°, 72°, ...）
    for i in range(10):
        angle = i * 36
        x_marker = np.cos(np.deg2rad(angle))
        y_marker = np.sin(np.deg2rad(angle))
        plt.plot(x_marker, y_marker, 'ko', markersize=8, alpha=0.3)
        plt.text(x_marker*1.1, y_marker*1.1, str(i), fontsize=12, ha='center', va='center')
    
    # 数字ごとに色を設定
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # 各数字からサンプルをピックアップ
    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        print(f"数字 {digit}: {len(digit_indices)} 個のサンプル")
        if len(digit_indices) == 0:
            continue
            
        # 数字ごとにランダムにサンプルを選択（固定シード）
        np.random.seed(42)  # 固定シードで同じサンプルを選択
        sample_indices = np.random.choice(digit_indices, min(num_samples, len(digit_indices)), replace=False)
        
        for i, idx in enumerate(sample_indices):
            # 正解位置（理想的な位置）
            true_angle = angles_test[idx]
            true_x = np.cos(np.deg2rad(true_angle))
            true_y = np.sin(np.deg2rad(true_angle))
            
            # 予測位置
            pred_angle = predicted_angles[idx]
            pred_x = np.cos(np.deg2rad(pred_angle))
            pred_y = np.sin(np.deg2rad(pred_angle))
            
            # 予測位置に画像を配置（少しオフセット）
            offset_radius = 0.7 + i * 0.03  # 複数サンプルが重ならないように
            img_x = pred_x * offset_radius
            img_y = pred_y * offset_radius
            
            # 画像を表示（予測位置）
            img = x_test[idx].reshape(28, 28)
            extent = [img_x - 0.06, img_x + 0.06, img_y - 0.06, img_y + 0.06]
            plt.imshow(img, extent=extent, cmap='gray', alpha=0.8)
            
            # 正解位置をマーク
            plt.plot(true_x, true_y, 'o', color=colors[digit], markersize=8, 
                    markeredgecolor='black', markeredgewidth=2, label=f'True {digit}' if i == 0 else "")
            
            # 予測位置をマーク
            plt.plot(pred_x, pred_y, 'x', color=colors[digit], markersize=10, 
                    markeredgewidth=3, label=f'Pred {digit}' if i == 0 else "")
            
            # 正解と予測を線で結ぶ
            plt.plot([true_x, pred_x], [true_y, pred_y], color=colors[digit], 
                    alpha=0.5, linewidth=1)
    
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)
    
    # タイトルにエポック情報を追加
    title = 'MNIST Digits on Unit Circle\n(Images at predicted positions, True: circles, Predicted: X)'
    if epoch is not None:
        title = f'Epoch {epoch}: {title}'
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 保存
    plt.tight_layout()
    filename = f'mnist_circle_epoch_{epoch}.png' if epoch is not None else 'mnist_circle_visualization.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()  # メモリ節約のため閉じる
    
    print(f"視覚化を保存しました: {filename}")


class EpochVisualizationCallback(callbacks.Callback):
    """エポックごとに予測結果を可視化するコールバック"""
    def __init__(self, x_test, angles_test, visualization_epochs=None, similarity_based=False):
        super().__init__()
        self.x_test = x_test
        self.angles_test = angles_test
        self.visualization_epochs = visualization_epochs or [1, 5, 10, 15, 20]
        self.similarity_based = similarity_based
        
    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        if epoch_num in self.visualization_epochs:
            print(f"\nエポック {epoch_num} の予測結果を可視化中...")
            
            # 予測実行
            predictions = self.model.predict(self.x_test, verbose=0)
            predictions = associated_points_on_circle(predictions)
            predicted_angles = points_2_angles(predictions)
            
            # 平均偏差計算
            deviation = prediction_mean_deviation(self.angles_test, predicted_angles)
            print(f"エポック {epoch_num} の平均偏差: {deviation:.2f}°")
            
            # 角度から数字を取得
            y_test_digits = [get_digit_from_angle(angle, self.similarity_based) for angle in self.angles_test]
            
            # 可視化
            visualize_digits_on_circle(
                self.x_test, y_test_digits, predictions, 
                self.angles_test, predicted_angles, epoch=epoch_num
            )


def main():
    parser = argparse.ArgumentParser(description='MNIST円形回帰学習')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='学習エポック数')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='バッチサイズ')
    parser.add_argument('--runs', '-r', type=int, default=1, help='実験回数')
    parser.add_argument('--visualize_epochs', '-v', nargs='+', type=int, 
                       default=[1, 5, 10, 15, 20], help='可視化するエポック')
    parser.add_argument('--step_by_step', '--sbs', action='store_true', 
                       help='段階的可視化を有効にする')
    parser.add_argument('--similarity_based', '--sim', action='store_true',
                       help='視覚的類似性に基づく角度配置を使用')
    # 再現性 / ログ用引数は削除
    
    args = parser.parse_args()
    
    print(f'MNIST円形回帰学習開始')
    print(f'全データ使用, エポック: {args.epochs}, 実行回数: {args.runs}')
    if args.step_by_step:
        print(f'段階的可視化: エポック {args.visualize_epochs}')
    if args.similarity_based:
        print('視覚的類似性に基づく角度配置を使用')
    print('=' * 50)
    
    mean_deviations = np.zeros(args.runs, dtype=np.float32)
    angle_accuracies = np.zeros(args.runs, dtype=np.float32)
    
    for run in range(args.runs):
        print(f'実行 {run + 1}/{args.runs}')
        
        # データ準備
        (x_train, x_val, x_test), (y_train, y_val, y_test), angles_test = prepare_mnist_data(args.similarity_based)
        
        print(f'データ形状: Train={x_train.shape}, Val={x_val.shape}, Test={x_test.shape}')
        
        # モデル作成
        model = create_simple_cnn(input_shape=(28, 28, 1))
        model.compile(
            optimizer='adam',
            loss=linear_dist_squared_loss
        )
        
        # モデル保存設定
        weights_dir = 'weights/mnist_cnn'
        os.makedirs(weights_dir, exist_ok=True)
        checkpoint_path = f'{weights_dir}/mnist_run_{run}.keras'
        
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True,
            verbose=1
        )
        
        # 常に新規学習（各 run を独立扱い）
        print('学習開始... (fresh init)')
        callbacks = [checkpoint_callback]
        if args.step_by_step:
            vis_callback = EpochVisualizationCallback(x_test, angles_test, args.visualize_epochs, args.similarity_based)
            callbacks.append(vis_callback)
        history = model.fit(
            x_train, y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # 予測と評価
        predictions = model.predict(x_test, verbose=0)
        predictions = associated_points_on_circle(predictions)
        predicted_angles = points_2_angles(predictions)
        
        # 平均偏差計算
        deviation = prediction_mean_deviation(angles_test, predicted_angles)
        mean_deviations[run] = deviation
        
        # 角度ベース分類精度計算（±18°許容）
        angle_accuracy = calculate_angle_accuracy(predicted_angles, angles_test, tolerance=18.0)
        angle_accuracies[run] = angle_accuracy
        
        print(f'実行 {run + 1} の平均偏差: {deviation:.2f}°')
        print(f'実行 {run + 1} の角度分類精度 (±18°): {angle_accuracy*100:.2f}%')
        print('-' * 30)
        
        # 最終結果の視覚化（段階的でない場合）
        if run == 0 and not args.step_by_step:
            print("最終結果の視覚化を作成中...")
            y_test_digits = [get_digit_from_angle(angle, args.similarity_based) for angle in angles_test]
            visualize_digits_on_circle(x_test, y_test_digits, predictions, angles_test, predicted_angles)
    
    # 最終結果
    print('\n' + '=' * 50)
    print('最終結果')
    print('=' * 50)
    final_mean = np.mean(mean_deviations)
    final_std = np.std(mean_deviations)
    
    accuracy_mean = np.mean(angle_accuracies)
    accuracy_std = np.std(angle_accuracies)
    
    print(f'平均偏差: {final_mean:.2f} ± {final_std:.2f}°')
    print(f'角度分類精度 (±18°): {accuracy_mean*100:.2f} ± {accuracy_std*100:.2f}%')
    print(f'各実行の偏差: {mean_deviations}')
    print(f'各実行の精度: {[f"{a*100:.2f}%" for a in angle_accuracies]}')
    
    return final_mean, final_std


if __name__ == "__main__":
    main()