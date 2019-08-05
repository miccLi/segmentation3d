import numpy as np
import tensorflow as tf
Epsilon = 1e-12


def vnet(vol_in, labels, feature_channels=16, output_channels=2):
    
    # conv1
    conv1_channel = feature_channels
    conv1_1 = conv_layer(vol_in, channel_out=conv1_channel, name='conv1_1')
    conv1_1_norm = normalization(conv1_1, name='conv1_1_norm')
    conv1_temp = tf.concat([vol_in]*conv1_channel, axis=4)
    conv1_add = tf.nn.relu(conv1_1_norm + conv1_temp)
    
    # down_conv1
    conv2_channel = 2 * conv1_channel
    down_conv1 = down_conv_layer(conv1_add, channel_out=conv2_channel, name='down_conv1')
    down_conv1_norm = tf.nn.relu(normalization(down_conv1, name='down_conv1_norm'))
    
    # conv2
    conv2_1 = conv_layer(down_conv1_norm, channel_out=conv2_channel, name='conv2_1')
    conv2_1_norm = tf.nn.relu(normalization(conv2_1, name='conv2_1_norm'))
    conv2_2 = conv_layer(conv2_1_norm, channel_out=conv2_channel, name='conv2_2')
    conv2_2_norm = normalization(conv2_2, name='conv2_2_norm')
    conv2_add = tf.nn.relu(down_conv1_norm + conv2_2_norm)
    
    # down_conv2
    conv3_channel = 2 * conv2_channel
    down_conv2 = down_conv_layer(conv2_add, channel_out=conv3_channel, name='down_conv2')
    down_conv2_norm = tf.nn.relu(normalization(down_conv2, name='down_conv2_norm'))
    
    # conv3
    #conv3_1 = conv_layer(down_conv2_norm, channel_out=conv3_channel, name='conv3_1', kernel_size=(2,3,3))
    conv3_1 = conv_layer(down_conv2_norm, channel_out=conv3_channel, name='conv3_1')
    conv3_1_norm = tf.nn.relu(normalization(conv3_1, name='conv3_1_norm'))
    #conv3_2 = conv_layer(conv3_1_norm, channel_out=conv3_channel, name='conv3_2', kernel_size=(2,3,3))
    conv3_2 = conv_layer(conv3_1_norm, channel_out=conv3_channel, name='conv3_2')
    conv3_2_norm = tf.nn.relu(normalization(conv3_2, name='conv3_2_norm'))
    #conv3_3 = conv_layer(conv3_2_norm, channel_out=conv3_channel, name='conv3_3', kernel_size=(2,3,3))
    conv3_3 = conv_layer(conv3_2_norm, channel_out=conv3_channel, name='conv3_3')
    conv3_3_norm = normalization(conv3_3, name='conv3_3_norm')
    conv3_add = tf.nn.relu(down_conv2_norm + conv3_3_norm)
    
    # down_conv3
    conv4_channel = 2 * conv3_channel
    down_conv3 = down_conv_layer(conv3_add, channel_out=conv4_channel, name='down_conv3')
    down_conv3_norm = tf.nn.relu(normalization(down_conv3, name='down_conv3_norm'))
    
    # conv4
    #conv4_1 = conv_layer(down_conv3_norm, channel_out=conv4_channel, name='conv4_1', kernel_size=(1,3,3))
    conv4_1 = conv_layer(down_conv3_norm, channel_out=conv4_channel, name='conv4_1')
    conv4_1_norm = tf.nn.relu(normalization(conv4_1, name='conv4_1_norm'))
    #conv4_2 = conv_layer(conv4_1_norm, channel_out=conv4_channel, name='conv4_2', kernel_size=(1,3,3))
    conv4_2 = conv_layer(conv4_1_norm, channel_out=conv4_channel, name='conv4_2')
    conv4_2_norm = tf.nn.relu(normalization(conv4_2, name='conv4_2_norm'))
    #conv4_3 = conv_layer(conv4_2_norm, channel_out=conv4_channel, name='conv4_3', kernel_size=(1,3,3))
    conv4_3 = conv_layer(conv4_2_norm, channel_out=conv4_channel, name='conv4_3')
    conv4_3_norm = normalization(conv4_3, name='conv4_3_norm')
    conv4_add = tf.nn.relu(down_conv3_norm + conv4_3_norm)

    # down_conv4
    conv5_channel = 2 * conv4_channel
    down_conv4 = down_conv_layer(conv4_add, channel_out=conv5_channel, name='down_conv4', kernel_size=(1,2,2), stride=(1,2,2))
    down_conv4_norm = tf.nn.relu(normalization(down_conv4, name='down_conv4_norm'))

    # conv5
    conv5_1 = conv_layer(down_conv4_norm, channel_out=conv5_channel, name='conv5_1', kernel_size=(1, 3, 3))
    conv5_1_norm = tf.nn.relu(normalization(conv5_1, name='conv5_1_norm'))
    conv5_2 = conv_layer(conv5_1_norm, channel_out=conv5_channel, name='conv5_2', kernel_size=(1, 3, 3))
    conv5_2_norm = tf.nn.relu(normalization(conv5_2, name='conv5_2_norm'))
    conv5_3 = conv_layer(conv5_2_norm, channel_out=conv5_channel, name='conv5_3', kernel_size=(1, 3, 3))
    conv5_3_norm = normalization(conv5_3, name='conv5_3_norm')
    conv5_add = tf.nn.relu(down_conv4_norm + conv5_3_norm)

    # down_conv5
    conv6_channel = 2 * conv5_channel
    down_conv5 = down_conv_layer(conv5_add, channel_out=conv6_channel, name='down_conv5', kernel_size=(1, 2, 2),
                                 stride=(1, 2, 2))
    down_conv5_norm = tf.nn.relu(normalization(down_conv5, name='down_conv5_norm'))

    # conv6
    conv6_1 = conv_layer(down_conv5_norm, channel_out=conv6_channel, name='conv6_1', kernel_size=(1, 3, 3))
    conv6_1_norm = tf.nn.relu(normalization(conv6_1, name='conv6_1_norm'))
    conv6_2 = conv_layer(conv6_1_norm, channel_out=conv6_channel, name='conv6_2', kernel_size=(1, 3, 3))
    conv6_2_norm = tf.nn.relu(normalization(conv6_2, name='conv6_2_norm'))
    conv6_3 = conv_layer(conv6_2_norm, channel_out=conv6_channel, name='conv6_3', kernel_size=(1, 3, 3))
    conv6_3_norm = normalization(conv6_3, name='conv6_3_norm')
    conv6_add = tf.nn.relu(down_conv5_norm + conv6_3_norm)

    # up_conv5
    up_conv5 = up_conv_layer(conv6_add, channel_out=conv5_channel, name='up_conv5', kernel_size=(1, 2, 2),
                             stride=(1, 2, 2))
    up_conv5_norm = tf.nn.relu(normalization(up_conv5, name='up_conv5_norm'))
    up_conv5_cat = tf.concat([conv5_add, up_conv5_norm], axis=4)

    # conv5_r
    conv5_r1 = conv_layer(up_conv5_cat, channel_out=conv5_channel * 2, name='conv5_r1', kernel_size=(1, 3, 3))
    conv5_r1_norm = tf.nn.relu(normalization(conv5_r1, name='conv5_r1_norm'))
    conv5_r2 = conv_layer(conv5_r1_norm, channel_out=conv5_channel * 2, name='conv5_r2', kernel_size=(1, 3, 3))
    conv5_r2_norm = tf.nn.relu(normalization(conv5_r2, name='conv5_r2_norm'))
    conv5_r3 = conv_layer(conv5_r2_norm, channel_out=conv5_channel * 2, name='conv5_r3', kernel_size=(1, 3, 3))
    conv5_r3_norm = normalization(conv5_r3, name='conv5_r3_norm')
    conv5_r_add = tf.nn.relu(up_conv5_cat + conv5_r3_norm)

    # up_conv4
    up_conv4 = up_conv_layer(conv5_r_add, channel_out=conv4_channel, name='up_conv4', kernel_size=(1,2,2), stride=(1,2,2))
    up_conv4_norm = tf.nn.relu(normalization(up_conv4, name='up_conv4_norm'))
    up_conv4_cat = tf.concat([conv4_add, up_conv4_norm], axis=4)

    # conv4_r
    #conv4_r1 = conv_layer(up_conv4_cat, channel_out=conv4_channel * 2, name='conv4_r1', kernel_size=(1, 3, 3))
    conv4_r1 = conv_layer(up_conv4_cat, channel_out=conv4_channel * 2, name='conv4_r1')
    conv4_r1_norm = tf.nn.relu(normalization(conv4_r1, name='conv4_r1_norm'))
    #conv4_r2 = conv_layer(conv4_r1_norm, channel_out=conv4_channel * 2, name='conv4_r2', kernel_size=(1, 3, 3))
    conv4_r2 = conv_layer(conv4_r1_norm, channel_out=conv4_channel * 2, name='conv4_r2')
    conv4_r2_norm = tf.nn.relu(normalization(conv4_r2, name='conv4_r2_norm'))
    #conv4_r3 = conv_layer(conv4_r2_norm, channel_out=conv4_channel * 2, name='conv4_r3', kernel_size=(1, 3, 3))
    conv4_r3 = conv_layer(conv4_r2_norm, channel_out=conv4_channel * 2, name='conv4_r3')
    conv4_r3_norm = normalization(conv4_r3, name='conv4_r3_norm')
    conv4_r_add = tf.nn.relu(up_conv4_cat + conv4_r3_norm)

    # up_conv3
    up_conv3 = up_conv_layer(conv4_r_add, channel_out=conv3_channel, name='up_conv3')
    up_conv3_norm = tf.nn.relu(normalization(up_conv3, name='up_conv3_norm'))
    up_conv3_cat = tf.concat([conv3_add, up_conv3_norm], axis=4)
    
    # conv3_r
    #conv3_r1 = conv_layer(up_conv3_cat, channel_out=conv3_channel*2, name='conv3_r1', kernel_size=(2,3,3))
    conv3_r1 = conv_layer(up_conv3_cat, channel_out=conv3_channel * 2, name='conv3_r1')
    conv3_r1_norm = tf.nn.relu(normalization(conv3_r1, name='conv3_r1_norm'))
    #conv3_r2 = conv_layer(conv3_r1_norm, channel_out=conv3_channel*2, name='conv3_r2', kernel_size=(2,3,3))
    conv3_r2 = conv_layer(conv3_r1_norm, channel_out=conv3_channel * 2, name='conv3_r2')
    conv3_r2_norm = tf.nn.relu(normalization(conv3_r2, name='conv3_r2_norm'))
    #conv3_r3 = conv_layer(conv3_r2_norm, channel_out=conv3_channel*2, name='conv3_r3', kernel_size=(2,3,3))
    conv3_r3 = conv_layer(conv3_r2_norm, channel_out=conv3_channel * 2, name='conv3_r3')
    conv3_r3_norm = normalization(conv3_r3, name='conv3_r3_norm')
    conv3_r_add = tf.nn.relu(up_conv3_cat + conv3_r3_norm)
    
    # up_conv2
    up_conv2 = up_conv_layer(conv3_r_add, channel_out=conv2_channel, name='up_conv2')
    up_conv2_norm = tf.nn.relu(normalization(up_conv2, name='up_conv2_norm'))
    up_conv2_cat = tf.concat([conv2_add, up_conv2_norm], axis=4)
    
    # conv2_r
    conv2_r1 = conv_layer(up_conv2_cat, channel_out=conv2_channel*2, name='conv2_r1')
    conv2_r1_norm = tf.nn.relu(normalization(conv2_r1, name='conv2_r1_norm'))
    conv2_r2 = conv_layer(conv2_r1_norm, channel_out=conv2_channel*2, name='conv2_r2')
    conv2_r2_norm = normalization(conv2_r2, name='conv2_r2_norm')
    conv2_r_add = tf.nn.relu(up_conv2_cat + conv2_r2_norm)
    
    # up_conv1
    up_conv1 = up_conv_layer(conv2_r_add, channel_out=conv1_channel, name='up_conv1')
    up_conv1_norm = tf.nn.relu(normalization(up_conv1, name='up_conv1_norm'))
    up_conv1_cat = tf.concat([conv1_add, up_conv1_norm], axis=4)
    
    # conv1_r
    conv1_r1 = conv_layer(up_conv1_cat, channel_out=conv1_channel*2, name='conv1_r1')
    conv1_r1_norm = normalization(conv1_r1, name='conv1_r1_norm')
    conv1_r_add = tf.nn.relu(up_conv1_cat + conv1_r1_norm)
    
    # logit
    logits = conv_layer(conv1_r_add, channel_out=output_channels, kernel_size=(1,1,1), name='logits', use_bias=False)
    prediction = conclusion(logits, labels, output_channels)
    
    return prediction, logits


def conv_layer(feature_in, channel_out, name, kernel_size=(5,5,5), dilation_rate=(1,1,1), use_bias=True):
    w_stddev = np.sqrt(2 / (kernel_size[-1] ** 5 * 16))
    with tf.variable_scope(name):
        return tf.layers.conv3d(inputs=feature_in,
                                filters=channel_out,
                                kernel_size=kernel_size,
                                padding='SAME',
                                dilation_rate=dilation_rate,
                                kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                bias_initializer=tf.initializers.constant(0.1),
                                use_bias=use_bias,
                                trainable=True)


def conv_layer_2d(feature_in, channel_out, name, kernel_size=(3,3), dilation_rate=(1,1), strides=(1,1), use_bias=True):
    w_stddev = np.sqrt(2 / (kernel_size[-1] ** 3 * 16))
    with tf.variable_scope(name):
        return tf.layers.conv2d(inputs=feature_in,
                                filters=channel_out,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding='SAME',
                                dilation_rate=dilation_rate,
                                kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                bias_initializer=tf.initializers.constant(0.1),
                                use_bias=use_bias,
                                trainable=True)


def down_conv_layer(feature_in, channel_out, name, kernel_size=(2,2,2), stride=(2,2,2)):
    w_stddev = np.sqrt(2 / (kernel_size[-1] ** 5 * 16))
    with tf.variable_scope(name):
        return tf.layers.conv3d(inputs=feature_in,
                                filters=channel_out,
                                kernel_size=kernel_size,
                                strides=stride,
                                padding='SAME',
                                kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                bias_initializer=tf.initializers.constant(0.1),
                                trainable=True)


def up_conv_layer(feature_in, channel_out, name, kernel_size=(2,2,2), stride=(2,2,2)):
    w_stddev = np.sqrt(2 / (kernel_size[-1] ** 5 * 16))
    with tf.variable_scope(name):
        return tf.layers.conv3d_transpose(inputs=feature_in,
                                          filters=channel_out,
                                          kernel_size=kernel_size,
                                          strides=stride,
                                          padding='SAME',
                                          kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                          bias_initializer=tf.initializers.constant(0.1),
                                          trainable=True)


def normalization(vol_in, name):

    Shape = vol_in.shape
    Para_shape = Shape[-1]
    Mean, Var = tf.nn.moments(vol_in, [1, 2, 3], keep_dims=True)
    with tf.variable_scope(name):
        Gamma = tf.get_variable(name='Gamma', shape=Para_shape, dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        Beta = tf.get_variable(name='Beta', shape=Para_shape, dtype=tf.float32,  initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=vol_in, mean=Mean, variance=Var, offset=Beta, scale=Gamma, variance_epsilon=Epsilon)


def sigmoid_normalization(vol_in, name):

    shape_in = vol_in.shape
    [_, _, _, _, channel_num] = shape_in.as_list()
    group_num = channel_num
    identity_init = tf.eye(channel_num, dtype=tf.float32)*12.0-6.0

    with tf.variable_scope(name):
        norm_weights = tf.get_variable(name='norm_weights',
                                       dtype=tf.float32,
                                       initializer=identity_init,
                                       trainable=True)
        gamma = tf.get_variable(name='gamma',
                                shape=shape_in[-1],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0),
                                trainable=True)
        beta = tf.get_variable(name='beta',
                               shape=shape_in[-1],
                               dtype=tf.float32,
                               initializer=tf.constant_initializer(0.0),
                               trainable=True)

    norm_weights_expand = tf.nn.sigmoid(tf.reshape(norm_weights, [1, 1, 1, 1, channel_num, group_num]))

    mean, _ = tf.nn.moments(vol_in, [0, 1, 2, 3], keep_dims=True)
    mean_expand = tf.tile(tf.reshape(mean, shape=[1, 1, 1, 1, channel_num, 1]), multiples=[1, 1, 1, 1, 1, group_num])
    mean_weighted = tf.reduce_sum(norm_weights_expand*mean_expand, axis=4)/tf.reduce_sum(norm_weights_expand, axis=4)

    var = tf.reduce_mean(tf.square(vol_in-mean_weighted), axis=[0, 1, 2, 3], keep_dims=True)

    return tf.nn.batch_normalization(x=vol_in, mean=mean_weighted, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)


def Layer_Normalization(vol_in, name):

    Shape = vol_in.shape
    Para_shape = Shape[-1]
    Mean, Var = tf.nn.moments(vol_in, [0, 1, 2, 3, 4], keep_dims=True)
    with tf.variable_scope(name):
        Gamma = tf.get_variable(name='Gamma', shape=Para_shape, dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        Beta = tf.get_variable(name='Beta', shape=Para_shape, dtype=tf.float32,  initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=vol_in, mean=Mean, variance=Var, offset=Beta, scale=Gamma, variance_epsilon=Epsilon)


def GN_Normalization(vol_in, name):

    Group = 4
    Shape_in = vol_in.shape
    Para_shape = Shape_in[-1]

    vol_in_reshape = tf.reshape(vol_in, shape=[Shape_in[0], Shape_in[1], Shape_in[2], Shape_in[3], Group, Shape_in[4]//Group])
    Mean_reshape, Var_reshape = tf.nn.moments(vol_in_reshape, [0, 1, 2, 3, 5], keep_dims=True)
    vol_in = tf.reshape((vol_in_reshape-Mean_reshape)/(Var_reshape+Epsilon), shape=Shape_in)

    Mean = tf.constant(value=0.0, dtype=tf.float32, shape=Shape_in)
    Var = tf.constant(value=1.0, dtype=tf.float32, shape=Shape_in)

    with tf.variable_scope(name):
        Gamma = tf.get_variable(name='Gamma', shape=Para_shape, dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        Beta = tf.get_variable(name='Beta', shape=Para_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=vol_in, mean=Mean, variance=Var, offset=Beta, scale=Gamma, variance_epsilon=0)


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
