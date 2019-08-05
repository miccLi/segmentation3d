import numpy as np
from collections import OrderedDict
import tensorflow as tf
Epsilon = 1e-12


def UNet3D(Input, Label, Unet_block, Kernel_size, Feature_num, N_class, Pool_size):
    
    Down_conv = OrderedDict()
    b_constant = 0.1
    w_stddev = np.sqrt(2 / (Kernel_size ** Unet_block * Feature_num))

    for i_layer in range(0, Unet_block):
        if i_layer == 0:
            Fout = Feature_num
        else:
            Fout = 2**i_layer*Feature_num

        Out_put_1 = tf.nn.relu(Normalization(Input=tf.layers.conv3d(inputs=Input, filters=Fout, kernel_size=Kernel_size, padding='SAME',
                                                                    kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                                                    bias_initializer=tf.initializers.constant(b_constant)),
                                             Name_scope='Downconv_%d_Conv_1' % (i_layer)))

        Down_conv[i_layer] = tf.nn.relu(Normalization(Input=tf.layers.conv3d(inputs=Out_put_1, filters=Fout, kernel_size=Kernel_size, padding='SAME',
                                                                             kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                                                             bias_initializer=tf.initializers.constant(b_constant)),
                                                      Name_scope='Downconv_%d_Conv_2' % (i_layer)))
        Input = tf.layers.max_pooling3d(inputs=Down_conv[i_layer], pool_size=Pool_size, strides=Pool_size)

    for i_layer in range(Unet_block, -1, -1):

        Fout = 2**i_layer*Feature_num

        Out_put_1 = tf.nn.relu(Normalization(Input=tf.layers.conv3d(inputs=Input, filters=Fout, kernel_size=Kernel_size, padding='SAME',
                                                                    kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                                                    bias_initializer=tf.initializers.constant(b_constant)),
                                             Name_scope='Upconv_%d_Conv_1' % (i_layer)))

        Out_put_2 = tf.nn.relu(Normalization(Input=tf.layers.conv3d(inputs=Out_put_1, filters=Fout, kernel_size=Kernel_size, padding='SAME',
                                                                    kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                                                    bias_initializer=tf.initializers.constant(b_constant)),
                                             Name_scope='Upconv_%d_Conv_2' % (i_layer)))
        
        if i_layer != 0:

            Out_put_3 = tf.nn.relu(Normalization(Input=tf.layers.conv3d_transpose(inputs=Out_put_2, filters=Fout//2,
                                                                                  kernel_size=Pool_size, strides=Pool_size, padding='VALID',
                                                                                  kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev),
                                                                                  bias_initializer=tf.initializers.constant(b_constant)),
                                                 Name_scope='Deconv_%d' % (i_layer)))
            Input = tf.concat([Down_conv[i_layer-1], Out_put_3], 4)
        else:
            Logit = tf.layers.conv3d(inputs=Out_put_2, filters=N_class, kernel_size=1, padding='SAME', use_bias=False,
                                     kernel_initializer=tf.initializers.truncated_normal(stddev=w_stddev))
    
    Prediction = Conclusion(Logit=Logit, Label=Label, N_class=N_class)

    return Prediction, Logit

def Instance_Normalization(Input, Name_scope):

    Shape = Input.shape
    Para_shape = Shape[-1]
    Mean, Var = tf.nn.moments(Input, [0, 1, 2, 3], keep_dims=True)
    with tf.variable_scope(Name_scope):
        Gamma = tf.get_variable(name='Gamma', shape=Para_shape, dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        Beta = tf.get_variable(name='Beta', shape=Para_shape, dtype=tf.float32,  initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=Input, mean=Mean, variance=Var, offset=Beta, scale=Gamma, variance_epsilon=Epsilon)

def Layer_Normalization(Input, Name_scope):

    Shape = Input.shape
    Para_shape = Shape[-1]
    Mean, Var = tf.nn.moments(Input, [0, 1, 2, 3, 4], keep_dims=True)
    with tf.variable_scope(Name_scope):
        Gamma = tf.get_variable(name='Gamma', shape=Para_shape, dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        Beta = tf.get_variable(name='Beta', shape=Para_shape, dtype=tf.float32,  initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=Input, mean=Mean, variance=Var, offset=Beta, scale=Gamma, variance_epsilon=Epsilon)

def GN_Normalization(Input, Name_scope):

    Group = 4
    Shape_in = Input.shape
    Para_shape = Shape_in[-1]

    Input_reshape = tf.reshape(Input, shape=[Shape_in[0], Shape_in[1], Shape_in[2], Shape_in[3], Group, Shape_in[4]//Group])
    Mean_reshape, Var_reshape = tf.nn.moments(Input_reshape, [0, 1, 2, 3, 5], keep_dims=True)
    Input = tf.reshape((Input_reshape-Mean_reshape)/(Var_reshape+Epsilon), shape=Shape_in)

    Mean = tf.constant(value=0.0, dtype=tf.float32, shape=Shape_in)
    Var = tf.constant(value=1.0, dtype=tf.float32, shape=Shape_in)

    with tf.variable_scope(Name_scope):
        Gamma = tf.get_variable(name='Gamma', shape=Para_shape, dtype=tf.float32, initializer=tf.constant_initializer(1.0))
        Beta = tf.get_variable(name='Beta', shape=Para_shape, dtype=tf.float32, initializer=tf.constant_initializer(0.0))

    return tf.nn.batch_normalization(x=Input, mean=Mean, variance=Var, offset=Beta, scale=Gamma, variance_epsilon=0)

def Normalization(Input, Name_scope):

    shape_in = Input.shape
    [_, _, _, _, channel_num] = shape_in.as_list()
    group_num = channel_num
    identity_init = tf.eye(channel_num, dtype=tf.float32)*12.0-6.0

    with tf.variable_scope(Name_scope):
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
    tf.summary.histogram('norm_weights', values=norm_weights)

    norm_weights_expand = tf.nn.sigmoid(tf.reshape(norm_weights, [1, 1, 1, 1, channel_num, group_num]))

    mean, _ = tf.nn.moments(Input, [0, 1, 2, 3], keep_dims=True)
    mean_expand = tf.tile(tf.reshape(mean, shape=[1, 1, 1, 1, channel_num, 1]), multiples=[1, 1, 1, 1, 1, group_num])
    mean_weighted = tf.reduce_sum(norm_weights_expand*mean_expand, axis=4)/tf.reduce_sum(norm_weights_expand, axis=4)

    var = tf.reduce_mean(tf.square(Input-mean_weighted), axis=[0, 1, 2, 3], keep_dims=True)

    return tf.nn.batch_normalization(x=Input, mean=mean_weighted, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)

def Clip_Normalization(Input, Name_scope):

    shape_in = Input.shape
    [_, _, _, _, channel_num] = shape_in.as_list()
    group_num = channel_num

    with tf.variable_scope(Name_scope):
        norm_weights = tf.get_variable(name='norm_weights',
                                       shape=[channel_num, group_num],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.5),
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
    tf.summary.histogram('norm_weights', values=norm_weights)

    norm_weights_expand = tf.clip_by_value(t=tf.reshape(norm_weights, [1, 1, 1, 1, channel_num, group_num]),
                                           clip_value_min=0.0, clip_value_max=1.0)

    mean, _ = tf.nn.moments(Input, [0, 1, 2, 3], keep_dims=True)
    mean_expand = tf.tile(tf.reshape(mean, shape=[1, 1, 1, 1, channel_num, 1]), multiples=[1, 1, 1, 1, 1, group_num])
    mean_weighted = tf.reduce_sum(norm_weights_expand*mean_expand, axis=4)/tf.reduce_sum(norm_weights_expand, axis=4)

    var = tf.reduce_mean(tf.square(Input-mean_weighted), axis=[0, 1, 2, 3], keep_dims=True)

    return tf.nn.batch_normalization(x=Input, mean=mean_weighted, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)

def Softmax_Normalization(Input, Name_scope):

    shape_in = Input.shape
    [_, _, _, _, channel_num] = shape_in.as_list()
    group_num = channel_num
    identity_init = tf.eye(channel_num, dtype=tf.float32)*20.0-10.0

    with tf.variable_scope(Name_scope):
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

    norm_weights_expand = tf.nn.softmax(tf.reshape(norm_weights, [1, 1, 1, 1, channel_num, group_num]), axis=4)

    mean, _ = tf.nn.moments(Input, [0, 1, 2, 3], keep_dims=True)
    mean_expand = tf.tile(tf.reshape(mean, shape=[1, 1, 1, 1, channel_num, 1]), multiples=[1, 1, 1, 1, 1, group_num])
    mean_weighted = tf.reduce_sum(norm_weights_expand*mean_expand, axis=4)

    var = tf.reduce_mean(tf.square(Input-mean_weighted), axis=[0, 1, 2, 3], keep_dims=True)

    return tf.nn.batch_normalization(x=Input, mean=mean_weighted, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)


def For_Feature_Normalization(Input, Name_scope):

    shape_in = Input.shape
    [_, depth, height, width, channel_num] = shape_in.as_list()

    with tf.variable_scope(Name_scope):
        norm_weights = tf.get_variable(name='norm_weights',
                                       shape=[channel_num, channel_num],
                                       dtype=tf.float32,
                                       initializer=tf.constant_initializer(1.0/channel_num),
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
    norm_weights = tf.nn.softmax(norm_weights, axis=1)

    feature_out_list = []
    for c in range(channel_num):
        print(c)
        channel_weight = norm_weights[c, :]
        channel_weight = tf.reshape(channel_weight, [1, 1, 1, 1, -1])
        channel_weight = tf.tile(channel_weight, multiples=[1, depth, height, width, 1])

        feature_slice = tf.slice(Input, begin=[0, 0, 0, 0, 0], size=[1, -1, -1, -1, -1])

        mean, var = tf.nn.weighted_moments(feature_slice, [0, 1, 2, 3, 4], frequency_weights=channel_weight,
                                           keep_dims=True)

        feature_out_slice = tf.slice(Input, begin=[0, 0, 0, 0, c], size=[1, -1, -1, -1, 1])
        feature_out_slice = (feature_out_slice - mean) / tf.sqrt(var + Epsilon)
        feature_out_list.append(feature_out_slice)

    feature_out = tf.concat(feature_out_list, axis=4)
    mean_out = tf.constant(value=0.0, dtype=tf.float32, shape=shape_in)
    var_out = tf.constant(value=1.0, dtype=tf.float32, shape=shape_in)

    return tf.nn.batch_normalization(x=feature_out, mean=mean_out, variance=var_out, offset=beta, scale=gamma,
                                     variance_epsilon=0)

def For_Mean_Normalization(Input, Name_scope):

    shape_in = Input.shape
    channel_num = shape_in[-1]

    with tf.variable_scope(Name_scope):
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

    mean, var = tf.nn.moments(Input, [0, 1, 2, 3])

    mean_weighted = []
    for c in range(shape_in[-1]):
        with tf.variable_scope(Name_scope+'_%d'%(c)):
            norm_weights = tf.get_variable(name='norm_weights',
                                           shape=[shape_in[-1]],
                                           dtype=tf.float32,
                                           initializer=tf.constant_initializer(0.0),
                                           trainable=True)
        mean_weighted.append(tf.reduce_mean(tf.nn.softmax(norm_weights)*mean))

    mean_weighted_reshape = tf.reshape(mean_weighted, shape=[1, 1, 1, 1, channel_num])

    return tf.nn.batch_normalization(x=Input, mean=mean_weighted_reshape, variance=var, offset=beta, scale=gamma,
                                     variance_epsilon=Epsilon)

def Conclusion(Logit, Label, N_class):

    foreground = tf.cast(x=tf.greater_equal(x=tf.slice(input_=tf.nn.softmax(logits=Logit), begin=[0, 0, 0, 0, 0],
                                                       size=[-1, -1, -1, -1, N_class-1]),
                                            y=0.5),
                         dtype=tf.int32)
    background = tf.subtract(x=1, y=foreground)
    classes = tf.concat(values=[foreground, background], axis=4)

    foreground_gt = tf.cast(x=tf.slice(input_=Label, begin=[0, 0, 0, 0, 0], size=[-1, -1, -1, -1, 1]), dtype=tf.int32)
    And = tf.bitwise.bitwise_and(foreground, foreground_gt)
    Or = tf.bitwise.bitwise_or(foreground, foreground_gt)

    IoU_1 = tf.constant(1, dtype=tf.float32)
    IoU_2 = tf.cast(tf.reduce_sum(And), dtype=tf.float32) / tf.cast(tf.reduce_sum(Or), dtype=tf.float32)
    IoU = tf.cond(tf.equal(tf.reduce_sum(Or), 0), lambda: IoU_1, lambda: IoU_2)

    Prediction = {"probabilities": tf.nn.softmax(logits=Logit, name="probabilities"),
                  "classes": classes,
                  'IoU': IoU,
                  'Or': tf.reduce_sum(Or),
                  'And': tf.reduce_sum(And)}

    return Prediction