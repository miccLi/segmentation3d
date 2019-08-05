import numpy as np
from collections import OrderedDict
import tensorflow as tf

Epsilon = 1e-12


def unet3d(vol_in, labels, downsampling=3, feature_channels=16, output_channels=2, pool_size=2, kernel_size=3):
    layers = OrderedDict()
    b_constant = 0.1
    w_stddev = np.sqrt(2 / (kernel_size ** downsampling * feature_channels))

    for i_layer in range(0, downsampling):
        if i_layer == 0:
            channel_out = feature_channels
        else:
            channel_out = 2 ** i_layer * feature_channels

        feature_out_1 = tf.nn.relu(
            normalization(
                vol_in=tf.layers.conv3d(inputs=vol_in, filters=channel_out, kernel_size=kernel_size, padding='SAME',
                                        kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                        bias_initializer=tf.initializers.constant(b_constant)),
                name='Downconv_%d_Conv_1' % (i_layer)))

        layers[i_layer] = tf.nn.relu(normalization(
            vol_in=tf.layers.conv3d(inputs=feature_out_1, filters=channel_out, kernel_size=kernel_size, padding='SAME',
                                    kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                    bias_initializer=tf.initializers.constant(b_constant)),
            name='Downconv_%d_Conv_2' % i_layer))
        vol_in = tf.layers.max_pooling3d(inputs=layers[i_layer], pool_size=pool_size, strides=pool_size)

    for i_layer in range(downsampling, -1, -1):

        channel_out = 2 ** i_layer * feature_channels

        feature_out_1 = tf.nn.relu(
            normalization(
                vol_in=tf.layers.conv3d(inputs=vol_in, filters=channel_out, kernel_size=kernel_size, padding='SAME',
                                        kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                        bias_initializer=tf.initializers.constant(b_constant)),
                name='Upconv_%d_Conv_1' % i_layer))

        feature_out_2 = tf.nn.relu(normalization(
            vol_in=tf.layers.conv3d(inputs=feature_out_1, filters=channel_out, kernel_size=kernel_size, padding='SAME',
                                    kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                    bias_initializer=tf.initializers.constant(b_constant)),
            name='Upconv_%d_Conv_2' % i_layer))

        if i_layer != 0:

            feature_out_3 = tf.nn.relu(
                normalization(vol_in=tf.layers.conv3d_transpose(inputs=feature_out_2, filters=channel_out // 2,
                                                                kernel_size=pool_size,
                                                                strides=pool_size, padding='VALID',
                                                                kernel_initializer=tf.initializers.truncated_normal(
                                                                    stddev=w_stddev),
                                                                bias_initializer=tf.initializers.constant(
                                                                    b_constant)),
                              name='Deconv_%d' % i_layer))
            vol_in = tf.concat([layers[i_layer - 1], feature_out_3], 4)
        else:
            logits = tf.layers.conv3d(inputs=feature_out_2, filters=output_channels, kernel_size=1, padding='SAME',
                                      use_bias=False,
                                      kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev))

    prediction = conclusion(logits=logits, labels=labels, output_channels=output_channels)

    return prediction, logits


def batch_normalization(vol_in, name):
    feature_shape = vol_in.shape
    n_channels = feature_shape[-1]
    mean, var = tf.nn.moments(vol_in, [0, 1, 2, 3], keep_dims=True)
    with tf.variable_scope(name):
        gamma = tf.get_variable(name='Gamma', shape=n_channels, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(name='Beta', shape=n_channels, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=vol_in, mean=mean, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)


def normalization(vol_in, name):
    feature_shape = vol_in.shape
    n_channels = feature_shape[-1]
    mean, var = tf.nn.moments(vol_in, [1, 2, 3], keep_dims=True)
    with tf.variable_scope(name):
        gamma = tf.get_variable(name='Gamma', shape=n_channels, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(name='Beta', shape=n_channels, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=vol_in, mean=mean, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)


def layer_normalization(vol_in, name):
    feature_shape = vol_in.shape
    n_channels = feature_shape[-1]
    mean, var = tf.nn.moments(vol_in, [0, 1, 2, 3, 4], keep_dims=True)
    with tf.variable_scope(name):
        gamma = tf.get_variable(name='Gamma', shape=n_channels, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(name='Beta', shape=n_channels, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=vol_in, mean=mean, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)


def group_normalization(vol_in, name):
    group = 4
    feature_shape_in = vol_in.shape
    n_channels = feature_shape_in[-1]

    vol_in_reshape = tf.reshape(vol_in,
                                shape=[feature_shape_in[0], feature_shape_in[1], feature_shape_in[2],
                                       feature_shape_in[3], group, feature_shape_in[4] // group])
    mean_reshape, var_reshape = tf.nn.moments(vol_in_reshape, [0, 1, 2, 3, 5], keep_dims=True)
    vol_in = tf.reshape((vol_in_reshape - mean_reshape) / (var_reshape + Epsilon), shape=feature_shape_in)

    mean = tf.constant(value=0.0, dtype=tf.float32, shape=feature_shape_in)
    var = tf.constant(value=1.0, dtype=tf.float32, shape=feature_shape_in)

    with tf.variable_scope(name):
        gamma = tf.get_variable(name='Gamma', shape=n_channels, dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable(name='Beta', shape=n_channels, dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=vol_in, mean=mean, variance=var, offset=beta, scale=gamma, variance_epsilon=0)


def sigmoid_normalization(vol_in, name):
    shape_in = vol_in.shape
    [_, _, _, _, channel_num] = shape_in.as_list()
    group_num = channel_num
    identity_init = tf.eye(channel_num, dtype=tf.float32) * 12.0 - 6.0

    with tf.variable_scope(name):
        norm_weights = tf.get_variable(name='norm_weights',
                                       dtype=tf.float32,
                                       initializer=identity_init,
                                       trainable=True)
        gamma = tf.get_variable(name='Gamma',
                                shape=shape_in[-1],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0),
                                trainable=True)
        beta = tf.get_variable(name='Beta',
                               shape=shape_in[-1],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)
    tf.summary.histogram('norm_weights', values=norm_weights)

    norm_weights_expand = tf.nn.sigmoid(tf.reshape(norm_weights, [1, 1, 1, 1, channel_num, group_num]))

    mean, _ = tf.nn.moments(vol_in, [0, 1, 2, 3], keep_dims=True)
    mean_expand = tf.tile(tf.reshape(mean, shape=[1, 1, 1, 1, channel_num, 1]), multiples=[1, 1, 1, 1, 1, group_num])
    mean_weighted = tf.reduce_sum(norm_weights_expand * mean_expand, axis=4) / tf.reduce_sum(norm_weights_expand,
                                                                                             axis=4)

    var = tf.reduce_mean(tf.square(vol_in - mean_weighted), axis=[0, 1, 2, 3], keep_dims=True)

    return tf.nn.batch_normalization(x=vol_in, mean=mean_weighted, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)


def conclusion(logits, labels, output_channels):
    foreground = tf.cast(
        tf.greater_equal(
            tf.slice(tf.nn.softmax(logits=logits),
                     begin=[0, 0, 0, 0, 1],
                     size=[-1, -1, -1, -1, output_channels - 1]),
            y=0.5),
        dtype=tf.int32)
    background = tf.subtract(
        x=1, y=tf.cast(
            tf.reduce_any(
                tf.cast(foreground, dtype=tf.bool),
                axis=4,
                keepdims=True),
            dtype=tf.int32))
    classes = tf.concat(values=[background, foreground], axis=4)

    foreground_gt = tf.cast(
        tf.slice(labels,
                 begin=[0, 0, 0, 0, 1],
                 size=[-1, -1, -1, -1, output_channels - 1]),
        dtype=tf.int32)
    intersection = tf.bitwise.bitwise_and(foreground, foreground_gt)
    union = tf.bitwise.bitwise_or(foreground, foreground_gt)

    iou_list = []
    iou_1 = tf.constant(1, dtype=tf.float32)
    for c in range(output_channels - 1):
        i = tf.slice(intersection, begin=[0, 0, 0, 0, c], size=[-1, -1, -1, -1, 1])
        u = tf.slice(union, begin=[0, 0, 0, 0, c], size=[-1, -1, -1, -1, 1])
        iou_2 = tf.cast(tf.reduce_sum(i), dtype=tf.float32) / tf.cast(tf.reduce_sum(u), dtype=tf.float32)
        iou = tf.cond(tf.equal(tf.reduce_sum(u), 0), lambda: iou_1, lambda: iou_2)
        iou_list.append(iou)
    iou_2 = tf.cast(tf.reduce_sum(intersection), dtype=tf.float32) / tf.cast(tf.reduce_sum(union), dtype=tf.float32)
    iou = tf.cond(tf.equal(tf.reduce_sum(union), 0), lambda: iou_1, lambda: iou_2)

    prediction = {"probabilities": tf.nn.softmax(logits=logits, name="probabilities"),
                  "masks": classes,
                  'Or': tf.reduce_sum(union, axis=[0, 1, 2, 3]),
                  'And': tf.reduce_sum(intersection, axis=[0, 1, 2, 3]),
                  'IoU_list': iou_list,
                  'IoU': iou}

    return prediction
