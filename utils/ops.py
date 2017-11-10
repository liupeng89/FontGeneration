# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
import tensorflow as tf
import numpy as np

seed = 100


def batch_norm(x, is_training, epsilon=1e-5, decay=0.9, scope="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=decay, updates_collections=None, epsilon=epsilon,
                                        scale=True, is_training=is_training, scope=scope)


def conv2d(x, output_filters, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="conv2d"):
    with tf.variable_scope(scope):
        shape = x.shape
        W = tf.get_variable('W', [kh, kw, shape[-1], output_filters],
                            initializer=tf.truncated_normal_initializer(stddev=stddev, seed=seed))
        Wconv = tf.nn.conv2d(x, W, strides=[1, sh, sw, 1], padding='SAME')

        biases = tf.get_variable('b', [output_filters], initializer=tf.constant_initializer(0.0))
        Wconv_plus_b = tf.reshape(tf.nn.bias_add(Wconv, biases), Wconv.get_shape())

        return Wconv_plus_b


def deconv2d(x, output_shape, kh=5, kw=5, sh=2, sw=2, stddev=0.02, scope="deconv2d"):
    with tf.variable_scope(scope):
        # filter : [height, width, output_channels, in_channels]
        input_shape = x.get_shape().as_list()
        W = tf.get_variable('W', [kh, kw, output_shape[-1], input_shape[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev, seed=seed))

        deconv = tf.nn.conv2d_transpose(x, W, output_shape=output_shape,
                                        strides=[1, sh, sw, 1])

        biases = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv_plus_b = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        return deconv_plus_b


# LeakyReLU uses up too much memory. tf.maximum(0.1 * x, x)
# def lrelu(x, leak=0.2):
#     return tf.maximum(x, leak * x)
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)


def fc(x, output_size, stddev=0.02, scope="fc"):
    with tf.variable_scope(scope):
        shape = x.get_shape().as_list()
        W = tf.get_variable("W", [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev, seed=seed))
        b = tf.get_variable("b", [output_size],
                            initializer=tf.constant_initializer(0.0))
        return tf.matmul(x, W) + b


# ssim implementation
def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value