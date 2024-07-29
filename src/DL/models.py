from keras import models, layers


def get_cnn_classification_model(input_shape, n_classes, summary=True):
    '''
    Returns a simple CNN Classification model with n_classes-neurons in last layer and a first layer with shape input_shape
    :param input_shape: shape of input layer
    :param n_classes: # neurons in last layer
    :param summary: True if summary should be printed
    :return: returns keras Model object
    '''
    input = layers.Input(input_shape)
    h = layers.Conv2D(16, 5, activation='relu')(input)
    h = layers.MaxPooling2D((2, 2))(h)
    h = layers.Conv2D(32, 3, activation='relu')(h)
    h = layers.MaxPooling2D((2, 2))(h)
    h = layers.Flatten()(h)
    h = layers.Dense(256, activation='relu')(h)
    h = layers.Dense(16, activation='relu')(h)
    output = layers.Dense(n_classes, activation='softmax')(h)

    model = models.Model(input, output)
    if summary:
        model.summary()
    return model


def get_cnn_regression_model(input_shape, output_size=1, activation='linear',
                             summary=True):
    '''
    Returns a simple CNN Regression model with inupt-layer of shape input_shape
    :param input_shape: shape of input layer
    :param output_size: # neurons for output layer
    :param activation: activation function for last neuron
    :param summary: True if summary should be printed
    :return: returns keras Model object
    '''
    input = layers.Input(input_shape)
    h = layers.Conv2D(16, 5, activation='relu')(input)
    h = layers.MaxPooling2D((2, 2))(h)
    h = layers.Conv2D(32, 3, activation='relu')(h)
    h = layers.MaxPooling2D((2, 2))(h)
    h = layers.Flatten()(h)
    h = layers.Dense(256, activation='relu')(h)
    h = layers.Dense(16, activation='relu')(h)
    output = layers.Dense(output_size, activation=activation)(h)

    model = models.Model(input, output)
    if summary:
        model.summary()
    return model
