#!/usr/bin/env python3
"""
Unified MNIST training script for both classification and circular regression.

Usage:
    # Classification (10-class softmax)
    python mnist_train.py --task classification --epochs 10 --runs 3

    # Circular Regression (10 digits on unit circle)
    python mnist_train.py --task regression --epochs 20 --similarity_based

    # With visualization (regression only)
    python mnist_train.py --task regression --visualize --visualize_epochs 1 5 10

Dataset: MNIST handwritten digits (28x28 grayscale)
"""
import os
import sys
import argparse
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.DL.metrics import prediction_mean_deviation
from src.DL.losses import linear_dist_squared_loss
from src.regression.utils.data_loaders import load_mnist_data, prepare_mnist_for_regression
from src.regression.utils.model_builders import create_mnist_classification_model, create_mnist_regression_model
from src.regression.utils.training_utils import setup_checkpoint
from src.regression.utils.format_gt import associated_points_on_circle, points_2_angles


def calculate_angle_accuracy(predicted_angles, true_angles, tolerance=18.0):
    """Calculate tolerance-based accuracy for circular regression"""
    diffs = np.abs(predicted_angles - true_angles)
    circular_diff = np.minimum(diffs, 360 - diffs)
    return float(np.mean(circular_diff <= tolerance))


def main():
    parser = argparse.ArgumentParser(description='MNIST unified training (classification/regression)')
    parser.add_argument('--task', choices=['classification', 'regression'], required=True,
                       help='Task type: classification (softmax) or regression (circular)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                       help='Training epochs (default: 10 for cls, 20 for reg)')
    parser.add_argument('--batch_size', '-b', type=int, default=64)
    parser.add_argument('--runs', '-r', type=int, default=1)
    parser.add_argument('--save_weights', action='store_true')
    # Regression-specific
    parser.add_argument('--similarity_based', '--sim', action='store_true',
                       help='(Regression only) Use visual similarity-based angle mapping')
    parser.add_argument('--visualize', action='store_true',
                       help='(Regression only) Visualize predictions at specific epochs')
    parser.add_argument('--visualize_epochs', '-v', nargs='+', type=int,
                       default=[1, 5, 10, 15, 20],
                       help='(Regression only) Epochs to visualize')
    args = parser.parse_args()

    is_regression = (args.task == 'regression')

    # Adjust default epochs for regression if not specified
    if is_regression and args.epochs == 10:
        args.epochs = 20

    print(f'=== MNIST {args.task} ===')
    print(f'Epochs: {args.epochs}, Batch: {args.batch_size}, Runs: {args.runs}')
    if is_regression and args.similarity_based:
        print('Using visual similarity-based angle mapping')
    if is_regression and args.visualize:
        print(f'Visualization at epochs: {args.visualize_epochs}')
    print('=' * 50)

    # Results storage
    all_results = []

    for run in range(args.runs):
        print(f'\nRun {run + 1}/{args.runs}')

        # Load and prepare data
        if is_regression:
            (x_train, x_val, x_test), (y_train, y_val, y_test), angles_test = \
                prepare_mnist_for_regression(args.similarity_based)
            model = create_mnist_regression_model(input_shape=(28, 28, 1))
            model.compile(optimizer='adam', loss=linear_dist_squared_loss)
            monitor, mode = 'val_loss', 'min'
        else:
            (x_train, y_train), (x_test, y_test) = load_mnist_data()
            x_train, x_val, y_train, y_val = train_test_split(
                x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            model = create_mnist_classification_model(input_shape=(28, 28, 1), num_classes=10)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            monitor, mode = 'val_accuracy', 'max'

        print(f'Data shapes: Train={x_train.shape}, Val={x_val.shape}, Test={x_test.shape}')

        # Callbacks
        callbacks_list = []
        if args.save_weights:
            subdir = 'mnist_cnn' if is_regression else 'mnist_cnn_cls'
            filename = f'mnist_{args.task}_run_{run}.keras'
            checkpoint = setup_checkpoint(f'weights/{subdir}', filename, monitor=monitor, mode=mode)
            callbacks_list.append(checkpoint)

        if is_regression and args.visualize:
            from src.regression.visualization.mnist_visualize import EpochVisualizationCallback
            vis_callback = EpochVisualizationCallback(
                x_test, angles_test, args.visualize_epochs, args.similarity_based
            )
            callbacks_list.append(vis_callback)

        # Train
        print('Training...')
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        # Evaluate
        if is_regression:
            predictions = model.predict(x_test, verbose=0)
            predictions = associated_points_on_circle(predictions)
            predicted_angles = points_2_angles(predictions)

            deviation = prediction_mean_deviation(angles_test, predicted_angles)
            angle_accuracy = calculate_angle_accuracy(predicted_angles, angles_test, tolerance=18.0)

            all_results.append(deviation)
            print(f'Run {run + 1}: Mean deviation={deviation:.2f}°, '
                  f'Angle accuracy (±18°)={angle_accuracy*100:.2f}%')

            # Final visualization (if not using step-by-step and only 1 run)
            if run == 0 and not args.visualize and args.runs == 1:
                from src.regression.visualization.mnist_visualize import (
                    visualize_digits_on_circle, get_digit_from_angle
                )
                print("Creating final visualization...")
                y_test_digits = [get_digit_from_angle(angle, args.similarity_based) for angle in angles_test]
                visualize_digits_on_circle(x_test, y_test_digits, predictions,
                                          angles_test, predicted_angles)

        else:
            # Load best weights if saved
            if args.save_weights:
                weights_path = Path(f'weights/mnist_cnn_cls/mnist_{args.task}_run_{run}.keras')
                if weights_path.exists():
                    print('Loading best weights for evaluation...')
                    model.load_weights(str(weights_path))

            test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
            all_results.append(test_acc)
            print(f'Run {run + 1}: Test accuracy={test_acc * 100:.2f}%')

            # Classification report for last run
            if run == args.runs - 1:
                y_pred_probs = model.predict(x_test, verbose=0)
                y_pred = np.argmax(y_pred_probs, axis=1)
                print('\nClassification report:')
                print(classification_report(y_test, y_pred, digits=4))

        print('-' * 40)

    # Summary
    print('\n=== Summary ===')
    if is_regression:
        print(f'Mean deviation: {np.mean(all_results):.2f} ± {np.std(all_results):.2f}°')
        print(f'Each run: {all_results}')
    else:
        print(f'Accuracies: {[f"{a*100:.2f}%" for a in all_results]}')
        print(f'Mean accuracy: {np.mean(all_results)*100:.2f} ± {np.std(all_results)*100:.2f}% (n={args.runs})')


if __name__ == "__main__":
    main()
