import tensorflow as tf
from keras import backend as K
import numpy as np
import json

def cyclic_loss(y_true, y_pred):
    '''
    Loss function when handeling degrees
    :param y_true: GT angle
    :param y_pred: Prediction
    :return: $\delta_{cyclic}(\alpha, \beta) \ = \
        \min(|\alpha - \beta|, 2\pi - |\beta - \alpha|)$
    '''
    a = tf.math.subtract(y_true, y_pred)
    b = tf.math.subtract(tf.math.add(y_true,  360), y_pred)
    clockwise_distance = K.switch(y_true >= y_pred, a, b)

    a = tf.math.subtract(y_pred, y_true)
    b = tf.math.subtract(tf.math.add(y_pred,  360), y_true)
    counter_clockwise_distance = K.switch(y_pred >= y_true, a, b)

    loss = K.switch(clockwise_distance < counter_clockwise_distance,
                    tf.math.abs(clockwise_distance),
                    tf.math.abs(counter_clockwise_distance))
    return loss


def cyclic_squared_loss(y_true, y_pred):
    '''
    Loss-function when handeling degrees
    :param y_true: GT angle
    :param y_pred: Prediction
    :return: $\delta_{cyclic}^2(\alpha, \beta) \ = \ (\delta_{cyclic}(\alpha, \beta))^2$
    '''
    loss = cyclic_loss(y_true, y_pred)
    squared = tf.math.square(loss)
    return squared


def linear_dist_loss(y_true, y_pred):
    '''
    Loss function calculates absolute difference, can be used for either angles or multidim points
    :param y_true: GT
    :param y_pred: Prediction
    :return: $\delta_{linear}(\alpha, \beta) \ = \ |\alpha-\beta|$ or
        \delta_{dist}(\Vec{x}, \Vec{y}) \ = \ |x_1-y_1|+|x_2-y_2|
        if used with the keras framework the output vector $\Vec{d} = \delta_{dist}(\Vec{x}, \Vec{y})$
        is then transformed to a real number \frac{1}{n} \sum_i^n{d_i}$
    '''
    return tf.math.abs(tf.math.subtract(y_true, y_pred))


def linear_dist_squared_loss(y_true, y_pred):
    '''
    Loss function squared difference, can be used for either angles or multidim points
    :param y_true: GT
    :param y_pred: Prediction
    :return: $\delta_{linear}^2(\alpha, \beta) \ = \ (\delta_{linear}(\alpha, \beta))^2$ or
        \delta_{dist}^2(\Vec{x}, \Vec{y}) \ = \ (\delta_{dist}(\Vec{x}, \Vec{y}))^2
        if used with the keras framework the output vector $\Vec{d} = \delta_{dist}(\Vec{x}, \Vec{y})$
        is then transformed to a real number \frac{1}{n} \sum_i^n{d_i}$
    '''
    loss = tf.math.abs(tf.math.subtract(y_true, y_pred))
    squared = tf.math.square(loss)
    return squared


def cyclic_cos_loss(y_true, y_pred):
    '''
    Loss function used für circular regression with angle output
    :param y_true: GT
    :param y_pred: Prediction
    :return: $\delta_{cos}(\alpha, \beta) \ = \ -\cos(\alpha-\beta)$
    '''
    return - tf.math.cos(tf.math.subtract(y_true, y_pred))


def eucl_dist_loss(y_true, y_pred):
    """
    Euclidean distance loss, can be used in context of circular regession for points on unit circle
    :param y_true: GT
    :param y_pred: Prediction
    :return: $\delta_{eucl}(\Vec{x}, \Vec{y}) \ = \ \sqrt{(x_1-y_1)^2+(x_2-y_2)^2}$
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


