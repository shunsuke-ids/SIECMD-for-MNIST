#!/usr/bin/env python3
"""
Unified model builders for Jurkat and MNIST experiments.

This module provides reusable functions to construct CNN backbones and task-specific heads
for both classification and circular regression tasks.
"""
from keras import models as km, layers as kl


def create_jurkat_backbone(input_shape=(66, 66, 1)):
    """
    Create CNN backbone for Jurkat (CellCycle) dataset.

    Architecture:
        Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool ->
        Conv2D(128) -> MaxPool -> Conv2D(256) -> GlobalAvgPool ->
        Dense(256) -> Dense(128)

    Args:
        input_shape: Input image shape (height, width, channels)

    Returns:
        inputs: Input layer
        features: Feature layer (128-dim)
    """
    inputs = kl.Input(shape=input_shape)
    x = kl.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = kl.MaxPooling2D()(x)
    x = kl.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = kl.GlobalAveragePooling2D()(x)
    x = kl.Dense(256, activation='relu')(x)
    features = kl.Dense(128, activation='relu', name='penultimate_features')(x)
    return inputs, features


def create_mnist_backbone(input_shape=(28, 28, 1)):
    """
    Create CNN backbone for MNIST dataset.

    Architecture:
        Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool ->
        Conv2D(128) -> GlobalAvgPool -> Dense(256) -> Dense(128)

    Args:
        input_shape: Input image shape (height, width, channels)

    Returns:
        inputs: Input layer
        features: Feature layer (128-dim)
    """
    inputs = kl.Input(shape=input_shape)
    x = kl.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = kl.MaxPooling2D((2, 2))(x)
    x = kl.Conv2D(64, (3, 3), activation='relu')(x)
    x = kl.MaxPooling2D((2, 2))(x)
    x = kl.Conv2D(128, (3, 3), activation='relu')(x)
    x = kl.GlobalAveragePooling2D()(x)
    x = kl.Dense(256, activation='relu')(x)
    features = kl.Dense(128, activation='relu', name='penultimate_features')(x)
    return inputs, features


def add_classification_head(features, num_classes):
    """
    Add classification head (softmax) to features.

    Args:
        features: Feature tensor from backbone
        num_classes: Number of output classes

    Returns:
        Output layer with softmax activation
    """
    return kl.Dense(num_classes, activation='softmax')(features)


def add_regression_head(features, activation):
    """
    Add circular regression head (2D output) to features.

    Args:
        features: Feature tensor from backbone
        activation: Activation function for output layer (e.g., sigmoid_activation)

    Returns:
        Output layer with 2 units (x, y coordinates on circle)
    """
    return kl.Dense(2, activation=activation)(features)


def create_jurkat_classification_model(input_shape=(66, 66, 1), num_classes=7):
    """
    Create complete Jurkat classification model.

    Args:
        input_shape: Input image shape
        num_classes: Number of classes

    Returns:
        Compiled Keras model
    """
    inputs, features = create_jurkat_backbone(input_shape)
    outputs = add_classification_head(features, num_classes)
    return km.Model(inputs, outputs)


def create_jurkat_regression_model(input_shape=(66, 66, 1), activation=None):
    """
    Create complete Jurkat circular regression model.

    Args:
        input_shape: Input image shape
        activation: Activation for regression head (default: sigmoid_activation)

    Returns:
        Compiled Keras model
    """
    if activation is None:
        from src.DL.activation_functions import sigmoid_activation
        activation = sigmoid_activation

    inputs, features = create_jurkat_backbone(input_shape)
    outputs = add_regression_head(features, activation)
    return km.Model(inputs, outputs)


def create_mnist_classification_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create complete MNIST classification model.

    Args:
        input_shape: Input image shape
        num_classes: Number of classes

    Returns:
        Compiled Keras model
    """
    inputs, features = create_mnist_backbone(input_shape)
    outputs = add_classification_head(features, num_classes)
    return km.Model(inputs, outputs)


def create_mnist_regression_model(input_shape=(28, 28, 1), activation=None):
    """
    Create complete MNIST circular regression model.

    Args:
        input_shape: Input image shape
        activation: Activation for regression head (default: sigmoid_activation)

    Returns:
        Compiled Keras model
    """
    if activation is None:
        from src.DL.activation_functions import sigmoid_activation
        activation = sigmoid_activation

    inputs, features = create_mnist_backbone(input_shape)
    outputs = add_regression_head(features, activation)
    return km.Model(inputs, outputs)

def create_jurkat_multitask_model(input_shape=(66, 66, 1), num_classes=7):
    """
    Create multi-task model with both regression and classification outputs.
    Uses the same backbone as single-task models.
    
    Args:
        input_shape: Input image shape
        num_classes: Number of classes for classification
        
    Returns:
        Model with two outputs: 'regression' and 'classification'
    """
    from src.DL.activation_functions import sigmoid_activation

    # 既存のバックボーンを使用
    inputs, features = create_jurkat_backbone(input_shape)

    # 回帰ヘッド（単位円上の2D座標、カスタムシグモイド活性化）
    regression_out = add_regression_head(features, sigmoid_activation)
    regression_out = kl.Lambda(lambda x: x, name='regression')(regression_out)

    # 分類ヘッド（7クラスのsoftmax）
    classification_out = add_classification_head(features, num_classes)
    classification_out = kl.Lambda(lambda x: x, name='classification')(classification_out)

    model = km.Model(inputs=inputs, outputs=[regression_out, classification_out])

    return model
