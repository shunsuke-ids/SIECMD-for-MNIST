import numpy as np
from keras import backend as K
import tensorflow as tf

def linear_activation(x):
    '''
    Activation function handeling degrees
    :param x: Real input value apply to final layer
    :return: $\varphi_{linear}(x) \ = \
        \begin{cases} 0 \,& x \leq 0 \\ min(x, 2\pi)  \,& \text{else}
        \end{cases}$
    '''
    return K.switch(x <= 0, tf.math.maximum(x, 0.), tf.math.minimum(x,  359))


def cyclic_activation(x):
    '''
    Activation function handeling degrees
    Taking into account that angle results are cyclic
    :param x: Real input value apply to final layer
    :return: $\varphi_{cyclic}(x) \ = \ x \bmod 2\pi$
    '''
    return tf.math.mod(x, 360)


def sigmoid_activation(x):
    '''
    Sigmoidal activation function that values are in [-1, 1]
    :param x: Real input value apply to final layer
    :return: $\varphi_{sigmoid}(x) \ = \ \frac{2e^x}{e^x+1}-1$
    '''
    a = tf.math.multiply(tf.math.exp(x), 2.)
    b = tf.math.add(tf.math.exp(x), 1.)

    return tf.math.subtract(tf.math.divide(a, b), 1.)
