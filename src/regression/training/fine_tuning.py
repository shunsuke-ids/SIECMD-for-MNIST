#!/usr/bin/env python3
"""
Fine-tuning script for circular regression using pre-trained models.

Purpose:
    Transfer learning approach using pre-trained backbones (EfficientNetV2, ResNet, YOLO)
    for circular regression tasks. This script demonstrates fine-tuning with frozen/unfrozen
    backbone layers.

Usage:
    python fine_tuning.py <root_dir> <data_dir> <dataset> [options]

Example:
    python fine_tuning.py /path/to/project /path/to/data mydataset --epochs 10

Note:
    This is distinct from probing.py which uses a simple CNN from scratch.
"""
import os
import pickle as pkl
import argparse

import keras_cv
from keras import models as km, layers as kl
from keras.callbacks import ModelCheckpoint

from src.DL.metrics import prediction_mean_deviation
from src.preprocessing.handle_dataset import *
from src.DL.activation_functions import *
from src.DL.losses import *

from src.regression.utils.format_gt import *
from src.regression.circular_operations import *

parser = argparse.ArgumentParser(description='Train circular regression model')
parser.add_argument('root_dir', help='Path to SIECMD Project')
parser.add_argument('data_dir', help='Path to dataset')
parser.add_argument('dataset', help='Name of dataset')

parser.add_argument('--n', '-n', type=int, default=4)
parser.add_argument('--epochs', '-e', type=int, default=10)
parser.add_argument('--fine_tuning_epochs', '-fte', type=int, default=50)
parser.add_argument('--batch_size', '-b', type=int, default=32)
parser.add_argument('--prediction_tolerance', '-pt', type=int, default=45)

args = parser.parse_args()

LOSS = linear_dist_squared_loss
ACTIVATION = sigmoid_activation
TRANSFER_LEARNING = False

backbone_name = 'efficientnetv2_s_imagenet'
last_conv = {'efficientnetv2_s_imagenet': 'top_activation', 'yolo_v8_xs_backbone_coco': 'stack4_spp_fast_output',
             'resnet50_v2_imagenet': 'post_relu'}

# backbone = keras_cv.models.YOLOV8Backbone.from_preset("yolo_v8_xs_backbone_coco", load_weights=True)
# backbone = keras_cv.models.ResNetV2Backbone.from_preset("resnet50_v2_imagenet", load_weights=True)
backbone = keras_cv.models.EfficientNetV2Backbone.from_preset(backbone_name, load_weights=True)

print(f'Results for Finetuning {backbone_name} (n={args.n}), [{args.epochs}, {args.fine_tuning_epochs}] Epochs\n\n')

mean_deviations = np.zeros(args.n, dtype=np.float32)
n_areas = 18
for i in range(0, args.n):
    print(f'{i + 1}.\n')

    with open(f'{args.data_dir}/{args.dataset}_{i}.pkl',
              'rb') as f:
        data = pkl.load(f)

    ((X_train, X_val, X_test), (y_train, y_val, y_test)) = data

    y_train = angles_2_unit_circle_points(y_train)
    y_val = angles_2_unit_circle_points(y_val)

    X_train = make_3_channel_imgs(X_train)
    X_val = make_3_channel_imgs(X_val)
    X_test = make_3_channel_imgs(X_test)

    # Construct custom head for pre-trained model
    h = kl.GlobalAveragePooling2D()(backbone.output)
    h = kl.Dense(1024, activation='relu')(h)
    h = kl.Dense(256, activation='relu')(h)
    output = kl.Dense(2, activation=ACTIVATION)(h)

    model = km.Model(backbone.input, output)

    weights_path = f'{args.root_dir}/weights'
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)

    weights_path = f'{weights_path}/fine_tuning'
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)

    weights_path = f'{weights_path}/{backbone_name}'
    if not os.path.isdir(weights_path):
        os.mkdir(weights_path)

    checkpoint_filepath = f'{weights_path}/weights{args.dataset}_{i}.keras'
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
        if os.path.isfile(checkpoint_filepath):
            model.load_weights(checkpoint_filepath)
        training_history = model.fit(X_train, y_train,
                                     epochs=args.epochs, batch_size=args.batch_size,
                                     validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback],
                                     verbose=1)
        # freeze backbone weights
        backbone.trainable = False

        training_history = model.fit(X_train, y_train,
                                     epochs=args.fine_tuning_epochs, batch_size=args.batch_size,
                                     validation_data=(X_val, y_val), callbacks=[model_checkpoint_callback],
                                     verbose=1)

    predictions = model.predict(X_test, verbose=0)
    predictions = associated_points_on_circle(predictions)
    predictions = points_2_angles(predictions)

    mean_deviations[i] = prediction_mean_deviation(y_test, predictions)
    print(f'\tdeviation: {mean_deviations[i]}')

print('\nFinished training')
mean_deviation = np.mean(mean_deviations)
std_diviation = np.sqrt(np.mean((mean_deviations - np.mean(mean_deviations)) ** 2))
print(f'\tMean diviation: {mean_deviation} +- {std_diviation}')
