#!/usr/bin/env python3
from keras import models as km, layers as kl

def create_mnist_backbone(input_shape=(28, 28, 1)):
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

def create_mnist_classification_model(input_shape=(28, 28, 1), num_classes=10):
    inputs, features = create_mnist_backbone(input_shape)
    outputs = kl.Dense(num_classes, activation='softmax')(features)
    return km.Model(inputs, outputs)

def create_mnist_regression_model(input_shape=(28, 28, 1)):
    from src.DL.activation_functions import sigmoid_activation

    inputs, features = create_mnist_backbone(input_shape)
    outputs = kl.Dense(2, activation=sigmoid_activation)(features)
    return km.Model(inputs, outputs)


def create_jurkat_backbone(input_shape=(66, 66, 1)):
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
    # x = kl.Dropout(0.3)(x)
    features = kl.Dense(128, activation='relu', name='penultimate_features')(x)
    return inputs, features

def create_jurkat_classification_model(input_shape=(66, 66, 1), num_classes=7):
    inputs, features = create_jurkat_backbone(input_shape)
    outputs = kl.Dense(num_classes, activation='softmax')(features)
    return km.Model(inputs, outputs)

def create_jurkat_regression_model(input_shape=(66, 66, 1)):
    from src.DL.activation_functions import sigmoid_activation

    inputs, features = create_jurkat_backbone(input_shape)
    outputs = kl.Dense(2, activation=sigmoid_activation)(features)
    return km.Model(inputs, outputs)

def create_jurkat_multitask_model(input_shape=(66, 66, 1), num_classes=7):
    from src.DL.activation_functions import sigmoid_activation

    inputs, features = create_jurkat_backbone(input_shape)

    regression_out = kl.Dense(2, activation=sigmoid_activation)(features)
    regression_out = kl.Lambda(lambda x: x, name='regression')(regression_out)

    classification_out = kl.Dense(num_classes, activation='softmax')(features)
    classification_out = kl.Lambda(lambda x: x, name='classification')(classification_out)

    model = km.Model(inputs=inputs, outputs=[regression_out, classification_out])

    return model


def create_phenocam_backbone(input_shape=(224, 224, 3)):
    inputs = kl.Input(shape=input_shape)
    x = kl.Conv2D(32, 3, padding='same')(inputs)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(64, 3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(128, 3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.MaxPooling2D()(x)

    x = kl.Conv2D(256, 3, padding='same')(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.GlobalAveragePooling2D()(x)

    x = kl.Dense(256)(x)
    x = kl.BatchNormalization()(x)
    x = kl.Activation('relu')(x)
    x = kl.Dropout(0.3)(x)

    x = kl.Dense(128)(x)
    x = kl.BatchNormalization()(x)
    features = kl.Activation('relu', name='penultimate_features')(x)

    return inputs, features

def create_phenocam_classification_model(input_shape=(224, 224, 3), num_classes=4):
    inputs, features = create_phenocam_backbone(input_shape)
    outputs = kl.Dense(num_classes, activation='softmax')(features)
    return km.Model(inputs, outputs)

def create_phenocam_regression_model(input_shape=(224, 224, 3)):
    from src.DL.activation_functions import sigmoid_activation

    inputs, features = create_phenocam_backbone(input_shape)
    outputs = kl.Dense(2, activation=sigmoid_activation)(features)
    return km.Model(inputs, outputs)

def create_phenocam_multitask_model(input_shape=(224, 224, 3), num_classes=4):
    from src.DL.activation_functions import sigmoid_activation

    inputs, features = create_phenocam_backbone(input_shape)

    regression_out = kl.Dense(2, activation=sigmoid_activation)(features)
    regression_out = kl.Lambda(lambda x: x, name='regression')(regression_out)

    classification_out = kl.Dense(num_classes, activation='softmax')(features)
    classification_out = kl.Lambda(lambda x: x, name='classification')(classification_out)

    model = km.Model(inputs=inputs, outputs=[regression_out, classification_out])

    return model