#!/usr/bin/env python3
"""
Visualization utilities for MNIST circular regression experiments.

This module provides functions to visualize MNIST digits on a unit circle,
showing both predicted and true angular positions.
"""
import numpy as np
import matplotlib.pyplot as plt
from keras import callbacks


def get_digit_from_angle(angle, similarity_based=False):
    """
    Get the corresponding digit from an angle.

    Args:
        angle: Angle in degrees (0-360)
        similarity_based: If True, use visual similarity-based mapping

    Returns:
        Digit (0-9)
    """
    if similarity_based:
        angle_to_digit = {
            0: 0, 36: 6, 72: 9, 108: 8, 144: 3,
            180: 2, 216: 5, 252: 1, 288: 7, 324: 4
        }
    else:
        angle_to_digit = {i * 36: i for i in range(10)}

    # Find the closest angle
    closest_angle = min(
        angle_to_digit.keys(),
        key=lambda x: min(abs(angle - x), abs(angle - x + 360), abs(angle - x - 360))
    )
    return angle_to_digit[closest_angle]


def visualize_digits_on_circle(x_test, y_test, predictions, angles_test, predicted_angles,
                                num_samples=5, epoch=None, save_path='mnist_circle_visualization.png'):
    """
    Plot MNIST digits on a unit circle showing predicted vs true positions.

    Args:
        x_test: Test images
        y_test: Test labels (digits)
        predictions: Predicted (x, y) coordinates on circle
        angles_test: True angles
        predicted_angles: Predicted angles
        num_samples: Number of samples to show per digit
        epoch: Optional epoch number for title
        save_path: Path to save the figure
    """
    plt.figure(figsize=(15, 15))

    # Convert y_test to numpy array if it's a list
    if isinstance(y_test, list):
        y_test = np.array(y_test)

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', alpha=0.5)
    plt.gca().add_patch(circle)

    # Draw angle markers (0°, 36°, 72°, ...)
    for i in range(10):
        angle = i * 36
        x_marker = np.cos(np.deg2rad(angle))
        y_marker = np.sin(np.deg2rad(angle))
        plt.plot(x_marker, y_marker, 'ko', markersize=8, alpha=0.3)
        plt.text(x_marker * 1.1, y_marker * 1.1, str(i), fontsize=12, ha='center', va='center')

    # Colors for each digit
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    # Sample images for each digit
    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        if len(digit_indices) == 0:
            continue

        # Randomly select samples (fixed seed for reproducibility)
        np.random.seed(42)
        sample_indices = np.random.choice(digit_indices, min(num_samples, len(digit_indices)), replace=False)

        for i, idx in enumerate(sample_indices):
            # True position
            true_angle = angles_test[idx]
            true_x = np.cos(np.deg2rad(true_angle))
            true_y = np.sin(np.deg2rad(true_angle))

            # Predicted position
            pred_angle = predicted_angles[idx]
            pred_x = np.cos(np.deg2rad(pred_angle))
            pred_y = np.sin(np.deg2rad(pred_angle))

            # Place image at predicted position (with slight offset to avoid overlap)
            offset_radius = 0.7 + i * 0.03
            img_x = pred_x * offset_radius
            img_y = pred_y * offset_radius

            # Display image
            img = x_test[idx].reshape(28, 28)
            extent = [img_x - 0.06, img_x + 0.06, img_y - 0.06, img_y + 0.06]
            plt.imshow(img, extent=extent, cmap='gray', alpha=0.8)

            # Mark true position
            plt.plot(true_x, true_y, 'o', color=colors[digit], markersize=8,
                    markeredgecolor='black', markeredgewidth=2, label=f'True {digit}' if i == 0 else "")

            # Mark predicted position
            plt.plot(pred_x, pred_y, 'x', color=colors[digit], markersize=10,
                    markeredgewidth=3, label=f'Pred {digit}' if i == 0 else "")

            # Connect true and predicted with a line
            plt.plot([true_x, pred_x], [true_y, pred_y], color=colors[digit],
                    alpha=0.5, linewidth=1)

    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, alpha=0.3)

    # Title
    title = 'MNIST Digits on Unit Circle\n(Images at predicted positions, True: circles, Predicted: X)'
    if epoch is not None:
        title = f'Epoch {epoch}: {title}'
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Visualization saved to: {save_path}")


class EpochVisualizationCallback(callbacks.Callback):
    """
    Keras callback to visualize predictions at specific epochs.
    """
    def __init__(self, x_test, angles_test, visualization_epochs=None, similarity_based=False):
        """
        Args:
            x_test: Test images
            angles_test: Test angles
            visualization_epochs: List of epochs to visualize (e.g., [1, 5, 10])
            similarity_based: Whether using similarity-based angle mapping
        """
        super().__init__()
        self.x_test = x_test
        self.angles_test = angles_test
        self.visualization_epochs = visualization_epochs or [1, 5, 10, 15, 20]
        self.similarity_based = similarity_based

    def on_epoch_end(self, epoch, logs=None):
        from src.regression.utils.format_gt import associated_points_on_circle, points_2_angles
        from src.DL.metrics import prediction_mean_deviation

        epoch_num = epoch + 1
        if epoch_num in self.visualization_epochs:
            print(f"\nVisualizing predictions at epoch {epoch_num}...")

            # Predict
            predictions = self.model.predict(self.x_test, verbose=0)
            predictions = associated_points_on_circle(predictions)
            predicted_angles = points_2_angles(predictions)

            # Calculate mean deviation
            deviation = prediction_mean_deviation(self.angles_test, predicted_angles)
            print(f"Epoch {epoch_num} mean deviation: {deviation:.2f}°")

            # Get digits from angles
            y_test_digits = [get_digit_from_angle(angle, self.similarity_based) for angle in self.angles_test]

            # Visualize
            visualize_digits_on_circle(
                self.x_test, y_test_digits, predictions,
                self.angles_test, predicted_angles,
                epoch=epoch_num,
                save_path=f'mnist_circle_epoch_{epoch_num}.png'
            )
