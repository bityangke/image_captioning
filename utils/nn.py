import math
import numpy as np
import tensorflow as tf

def weight(name, shape, init='he', range=1, stddev=0.33, init_val=None):
    if init_val is not None:
        initializer = tf.constant_initializer(init_val)
    elif init == 'uniform':
        initializer = tf.random_uniform_initializer(-range, range)
    elif init == 'normal':
        initializer = tf.random_normal_initializer(stddev = stddev)
    elif init == 'he':
        fan_in, _ = _get_dims(shape)
        std = math.sqrt(2.0 / fan_in)
        initializer = tf.random_normal_initializer(stddev = std)
    elif init == 'xavier':
        fan_in, fan_out = _get_dims(shape)
        range = math.sqrt(6.0 / (fan_in + fan_out))
        initializer = tf.random_uniform_initializer(-range, range)
    else:
        initializer = tf.truncated_normal_initializer(stddev = stddev)

    var = tf.get_variable(name, shape, initializer = initializer)
    tf.add_to_collection('l2', tf.nn.l2_loss(var))
    return var


def bias(name, dim, init_val=0.0):
    dims = dim if isinstance(dim, list) else [dim]
    return tf.get_variable(name, dims, initializer = tf.constant_initializer(init_val))


def batch_norm(x, name, is_train):
    with tf.variable_scope('batch_norm_'+name):
        inputs_shape = x.get_shape()
        axis = list(range(len(inputs_shape) - 1))
        param_shape = int(inputs_shape[-1])

        beta = tf.get_variable('beta', [param_shape], initializer = tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', [param_shape], initializer = tf.constant_initializer(1.0))
        batch_mean, batch_var = tf.nn.moments(x, axis)
        ema = tf.train.ExponentialMovingAverage(decay = 0.995)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_train, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


def nonlinear(x, nl=None):
    if nl == 'relu':
        return tf.nn.relu(x)
    elif nl == 'tanh':
        return tf.tanh(x)
    elif nl == 'sigmoid':
        return tf.sigmoid(x)
    else:
        return x


def fully_connected(x, output_size, name_w, name_b, is_train, bn=True, nl='relu', init_w='he', init_b=0):
    x_shape = _get_shape(x)
    w = weight(name_w, [x_shape[1], output_size], init_w)
    b = bias(name_b, [output_size], init_b)
    z = tf.nn.xw_plus_b(x, w, b)
    z = nonlinear(z, nl)
    if bn:
        z = batch_norm(z, name_w+'_'+name_b, is_train)
    return z


def fully_connected_no_bias(x, output_size, name_w, is_train, bn=True, nl='relu', init_w='he'):
    x_shape = _get_shape(x)
    w = weight(name_w, [x_shape[1], output_size], init_w)
    z = tf.matmul(x, w)
    z = nonlinear(z, nl)
    if bn:
        z = batch_norm(z, name_w, is_train)
    return z


def dropout(x, keep_prob, is_train):
    return tf.cond(is_train, lambda: tf.nn.dropout(x, keep_prob), lambda: x)


def _get_dims(shape):
    fan_in = np.prod(shape[:-1])
    fan_out = shape[-1]
    return fan_in, fan_out


def _get_shape(x):
    return [int(x.get_shape()[i]) for i in range(len(x.get_shape()))]

