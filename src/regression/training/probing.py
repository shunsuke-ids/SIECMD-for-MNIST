#!/usr/bin/env python3
"""
Probing (linear evaluation) script for circular regression using simple CNN.

Purpose:
    Train a simple CNN from scratch for circular regression tasks using
    cyclic activation and loss functions. Includes Test-Time Augmentation (TTA)
    for improved predictions.

Usage:
    python probing.py <root_dir> <data_dir> <dataset> [options]

Example:
    python probing.py /path/to/project /path/to/data mydataset --epochs 50

Note:
    This is distinct from fine_tuning.py which uses pre-trained models.
    TTA with n_rotations=4 is applied during inference for robustness.
"""
import argparse
import os
import pickle as pkl

from keras.callbacks import ModelCheckpoint

from src.preprocessing.augment import *
from src.regression.circular_operations import *
from src.DL.models import *
from src.DL.losses import *
from src.DL.activation_functions import *
from src.DL.metrics import *

parser = argparse.ArgumentParser(description='Train circular regression model')
parser.add_argument('root_dir', help='Path to SIECMD Project')
parser.add_argument('data_dir', help='Path to dataset')
parser.add_argument('dataset', help='Name of dataset')

parser.add_argument('--n', '-n', type=int, default=4)
parser.add_argument('--epochs', '-e', type=int, default=50)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--prediction_tolerance', '-pt', type=int, default=45)

args = parser.parse_args()

LOSS = cyclic_loss
ACTIVATION = cyclic_activation

print(f'Results for Training (n={args.n}), {args.epochs} Epochs\n\n')

mean_deviations = np.zeros(args.n, dtype=np.float32)

for i in range(args.n):
    print(f'{i + 1}.\n')
    with open(f'{args.data_dir}/{args.dataset}_{i}.pkl', 'rb') as f:
        data = pkl.load(f)

    ((X_train, X_val, X_test), (y_train, y_val, y_test)) = data

    model = get_cnn_regression_model(X_train.shape[1:], output_size=1, activation=ACTIVATION, summary=False)

    weights_path = f'{args.root_dir}/weights'
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)

    checkpoint_filepath = f'{weights_path}/weights_{i}.keras'

    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1)

    model.compile(optimizer='adam',
                  loss=LOSS)

    if os.path.isfile(checkpoint_filepath):
        print(f'Loads checkpoint file...\n{checkpoint_filepath}')
        model.load_weights(checkpoint_filepath)
    else:
        print(f'Starts training ...\n{checkpoint_filepath}')

        training_history = model.fit(X_train, y_train,
                                     epochs=args.epochs, batch_size=args.batch_size,
                                     validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback])

    # Test-time augmentation (TTA)
    n_rotations = 4
    X, y, rotations = TTA(X_test, y_test, n=n_rotations)
    pred = model.predict(X, verbose=0)
    pred = pred.reshape((len(pred)))
    pred = (pred - rotations) % 360

    predictions = np.zeros(y_test.shape)
    for j in range(int(pred.shape[0] / n_rotations)):
        values = pred[n_rotations * j: n_rotations * j + n_rotations]
        predictions[j] = np.round(circular_mean(values))

    mean_deviations[i] = prediction_mean_deviation(y_test, predictions)
    print(f'\tdeviation: {mean_deviations[i]}')

print('\nFinished training')
mean_deviation = np.mean(mean_deviations)
std_diviation = np.sqrt(np.mean((mean_deviations - np.mean(mean_deviations)) ** 2))
print(f'\tMean diviation: {mean_deviation} +- {std_diviation}')
