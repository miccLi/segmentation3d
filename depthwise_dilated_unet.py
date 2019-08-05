import numpy as np
import tensorflow as tf


class DDUNet:

    def __init__(self,
                 vol_in,
                 labels,
                 feature_channels=16,
                 output_channels=2,
                 downsampling=3,  # max: 9, recommend: 6?
                 downsampling_type='conv',  # 'conv', 'max_pooling'
                 upsampling_type='nearest_neighbour',  # 'deconv', 'nearest_neighbour', 'bilinear'
                 norm_type='IN'):
        # All layers
        self.layers = {}
        self.prediction = None
        self.logits = None

        # Network graph setup
        with tf.variable_scope('DDU_NET'):
            self.build_network(vol_in,
                               labels,
                               feature_channels,
                               output_channels,
                               downsampling,
                               downsampling_type,
                               upsampling_type,
                               norm_type)

    def build_network(self,
                      vol_in,
                      labels,
                      feature_channels=16,
                      output_channels=2,
                      downsampling=3,
                      downsampling_type='conv',  # 'conv', 'max-pooling'
                      upsampling_type='deconv',  # 'deconv', 'nearest neighbour', 'bilinear'
                      activation='relu',  # 'relu', 'prelu', 'elu'
                      norm_type='IN'):
        with tf.variable_scope('ENCODER'):
            for i in range(downsampling):
                if i == 0:
                    # Convolution 1-1
                    self.layers['conv1_1'] = self.conv_layer(vol_in=vol_in,
                                                             channel_out=feature_channels,
                                                             name='conv%d_1' % (i + 1),
                                                             kernel_size=(3, 7, 7),
                                                             norm=norm_type,
                                                             activation=activation)
                    # Downsampling
                    self.layers['conv1_downsampling'] = \
                        self.downsampling_layer(vol_in=self.layers['conv1_1'],
                                                downsampling_type=downsampling_type,
                                                norm=norm_type,
                                                name='conv%d_downsampling' % (i + 1))
                else:
                    # Convolution i-1
                    self.layers['conv%d_1' % (i + 1)] = self.conv_layer(vol_in=self.layers['conv%d_downsampling' % i],
                                                                        channel_out=feature_channels * 2 ** i,
                                                                        name='conv%d_1' % (i + 1),
                                                                        norm=norm_type,
                                                                        activation=activation)
                    # Convolution i-2
                    self.layers['conv%d_2' % (i + 1)] = self.conv_layer(vol_in=self.layers['conv%d_1' % (i + 1)],
                                                                        channel_out=feature_channels * 2 ** i,
                                                                        name='conv%d_2' % (i + 1),
                                                                        dilations=(3, 1, 1),
                                                                        norm=norm_type,
                                                                        activation=activation)
                    # Convolution i-3
                    # self.layers['conv%d_3' % (i + 1)] = self.conv_layer(vol_in=self.layers['conv%d_2' % (i + 1)],
                    #                                                     channel_out=feature_channels * 2 ** i,
                    #                                                     name='conv%d_3' % (i + 1),
                    #                                                     dilations=(3, 1, 1),
                    #                                                     norm=norm_type,
                    #                                                     activation=None)

                    # Residual
                    with tf.variable_scope('residual_%d' % (i + 1)):
                        self.layers['conv%d_res' % (i + 1)] = self.layers['conv%d_downsampling' % i] + \
                                                              self.layers['conv%d_2' % (i + 1)]
                        self.layers['conv%d_res' % (i + 1)] = \
                            self.activation_layer(self.layers['conv%d_res' % (i + 1)], activation)

                    # Downsampling
                    self.layers['conv%d_downsampling' % (i + 1)] = \
                        self.downsampling_layer(vol_in=self.layers['conv%d_res' % (i + 1)],
                                                downsampling_type=downsampling_type,
                                                name='conv%d_downsampling' % (i + 1))

        with tf.variable_scope('HIGH_LEVEL_FEATURES'):
            # High-level feature extraction
            self.layers['conv%d_1' % (downsampling + 1)] = self.conv_layer(
                vol_in=self.layers['conv%d_downsampling' % (i + 1)],
                channel_out=feature_channels * 2 ** downsampling,
                name='conv%d_1' % (downsampling + 1),
                norm=norm_type,
                activation=activation)
            self.layers['conv%d_2' % (downsampling + 1)] = self.conv_layer(
                vol_in=self.layers['conv%d_1' % (downsampling + 1)],
                channel_out=feature_channels * 2 ** downsampling,
                name='conv%d_2' % (downsampling + 1),
                dilations=(3, 1, 1),
                norm=norm_type,
                activation=activation)
            # self.layers['conv%d_3' % (downsampling + 1)] = self.conv_layer(
            #     vol_in=self.layers['conv%d_2' % (downsampling + 1)],
            #     channel_out=feature_channels * 2 ** downsampling,
            #     name='conv%d_3' % (downsampling + 1),
            #     norm=norm_type,
            #     dilations=(3, 1, 1),
            #     activation=None)

            # Residual
            with tf.variable_scope('residual_%d' % (downsampling + 1)):
                self.layers['conv%d_res' % (downsampling + 1)] = self.layers['conv%d_downsampling' % downsampling] + \
                                                                 self.layers['conv%d_2' % (downsampling + 1)]
                self.layers['conv%d_res' % (downsampling + 1)] = \
                    self.activation_layer(self.layers['conv%d_res' % (downsampling + 1)], activation)

        with tf.variable_scope('DECODER'):
            for i in range(downsampling - 1, 0, -1):
                # Upsampling and feature forwarding
                if i == downsampling - 1:
                    self.layers['conv%d_upsampling' % (i + 1)] = \
                        self.upsampling_layer(vol_in=self.layers['conv%d_res' % (downsampling + 1)],
                                              channel_out=feature_channels * 2 ** i,
                                              upsampling_type=upsampling_type,
                                              norm=norm_type,
                                              name='conv%d_upsampling' % (i + 1))
                else:
                    self.layers['conv%d_upsampling' % (i + 1)] = \
                        self.upsampling_layer(vol_in=self.layers['conv%d_d_res' % (i + 2)],
                                              channel_out=feature_channels * 2 ** i,
                                              upsampling_type=upsampling_type,
                                              norm=norm_type,
                                              name='conv%d_upsampling' % (i + 1))
                self.layers['conv%d_concat' % (i + 1)] = tf.concat([self.layers['conv%d_res' % (i + 1)],
                                                                    self.layers['conv%d_upsampling' % (i + 1)]],
                                                                   axis=4)

                # Convolution i-1
                self.layers['conv%d_d1' % (i + 1)] = self.conv_layer(vol_in=self.layers['conv%d_concat' % (i + 1)],
                                                                     channel_out=feature_channels * 2 ** (i + 1),
                                                                     name='conv%d_d1' % (i + 1),
                                                                     norm=norm_type,
                                                                     activation=activation)
                # Convolution i-2
                self.layers['conv%d_d2' % (i + 1)] = self.conv_layer(vol_in=self.layers['conv%d_d1' % (i + 1)],
                                                                     channel_out=feature_channels * 2 ** (i + 1),
                                                                     name='conv%d_d2' % (i + 1),
                                                                     dilations=(3, 1, 1),
                                                                     norm=norm_type,
                                                                     activation=activation)
                # Convolution i-3
                # self.layers['conv%d_d3' % (i + 1)] = self.conv_layer(vol_in=self.layers['conv%d_d2' % (i + 1)],
                #                                                      channel_out=feature_channels * 2 ** (i + 1),
                #                                                      name='conv%d_d3' % (i + 1),
                #                                                      norm=norm_type,
                #                                                      dilations=(3, 1, 1),
                #                                                      activation=None)

                # Residual for V-Net and atrous net
                with tf.variable_scope('residual_d%d' % (i + 1)):
                    self.layers['conv%d_d_res' % (i + 1)] = self.layers['conv%d_d2' % (i + 1)] + \
                                                            self.layers['conv%d_concat' % (i + 1)]
                    self.layers['conv%d_d_res' % (i + 1)] = self.activation_layer(self.layers['conv%d_d_res' % (i + 1)],
                                                                                  activation)

            self.layers['conv1_upsampling'] = self.upsampling_layer(vol_in=self.layers['conv2_d_res'],
                                                                    channel_out=feature_channels,
                                                                    upsampling_type=upsampling_type,
                                                                    norm=norm_type,
                                                                    name='conv1_upsampling')
            self.layers['conv1_concat'] = tf.concat([self.layers['conv1_1'],
                                                     self.layers['conv1_upsampling']],
                                                    axis=4)
            # self.layers['conv1_d1'] = self.conv_layer(vol_in=self.layers['conv1_concat'],
            #                                           channel_out=feature_channels,
            #                                           name='conv1_d1',
            #                                           norm=norm_type,
            #                                           activation=activation)

        self.logits = self.conv_layer(vol_in=self.layers['conv1_concat'],
                                      channel_out=output_channels,
                                      name='logits',
                                      kernel_size=(3, 7, 7),
                                      norm=None,
                                      activation=None,
                                      use_bias=False)
        self.layers['logits'] = self.logits
        self.prediction = self.result(self.logits, labels, output_channels)

    # Convolution with normalization and activation
    def conv_layer(self,
                   vol_in,
                   channel_out,
                   name,
                   kernel_size=(3, 3, 3),
                   strides=(1, 1, 1),
                   dilations=(1, 1, 1),
                   init=tf.initializers.variance_scaling(distribution='uniform',
                                                         scale=2.0,
                                                         mode='fan_in',
                                                         dtype=tf.float32),
                   norm='IN',
                   activation='relu',
                   use_bias=True):
        with tf.variable_scope(name):
            conv = tf.nn.conv3d(vol_in,
                                tf.get_variable('%s_weight' % name,
                                                shape=[kernel_size[0],
                                                       kernel_size[1],
                                                       kernel_size[2],
                                                       vol_in.shape[-1],
                                                       channel_out],
                                                dtype=tf.float32,
                                                initializer=init,
                                                trainable=True),
                                strides=(1, strides[0], strides[1], strides[2], 1),
                                padding='SAME',
                                dilations=(1, dilations[0], dilations[1], dilations[2], 1))
            if use_bias:
                conv = tf.nn.bias_add(conv,
                                      tf.get_variable('%s_bias' % name,
                                                      shape=[channel_out],
                                                      dtype=tf.float32,
                                                      initializer=tf.initializers.constant(0.1),
                                                      trainable=True))

            if norm:
                conv = self.normalization(conv, norm, '%s_norm' % name)

            if activation:
                conv = self.activation_layer(conv, activation)

            return conv

    # Convolution transpose with normalization and activation
    def deconv_layer(self,
                     vol_in,
                     channel_out,
                     name,
                     kernel_size=(3, 2, 2),
                     strides=(1, 2, 2),
                     norm='IN',
                     activation='prelu',
                     use_bias=True):
        input_shape = vol_in.get_shape().as_list()
        with tf.variable_scope(name):
            conv = tf.nn.conv3d_transpose(vol_in,
                                          tf.get_variable('%s_weight' % name,
                                                          shape=[kernel_size[0],
                                                                 kernel_size[1],
                                                                 kernel_size[2],
                                                                 channel_out,
                                                                 input_shape[-1]],
                                                          dtype=tf.float32,
                                                          initializer=tf.initializers.variance_scaling(
                                                              distribution='uniform',
                                                              scale=2.0,
                                                              mode='fan_in',
                                                              dtype=tf.float32),
                                                          trainable=True),
                                          output_shape=tf.TensorShape([input_shape[0],
                                                                       input_shape[1] * strides[0],
                                                                       input_shape[2] * strides[1],
                                                                       input_shape[3] * strides[2],
                                                                       channel_out]),
                                          strides=[1, strides[0], strides[1], strides[2], 1],
                                          padding='SAME')
            if use_bias:
                conv = tf.nn.bias_add(conv,
                                      tf.get_variable('%s_bias' % name,
                                                      shape=[channel_out],
                                                      dtype=tf.float32,
                                                      initializer=tf.initializers.constant(0.1),
                                                      trainable=True))

            if norm:
                conv = self.normalization(conv, norm, '%s_norm' % name)

            if activation:
                conv = self.activation_layer(conv, activation)

            return conv

    # Resize height and width in 3D volume
    def resize_layer(self, vol_in, scale, method):
        with tf.variable_scope('resize'):
            vol_shape = vol_in.shape
            vol_reshape = tf.reshape(vol_in,
                                     [vol_shape[0] * vol_shape[1], vol_shape[2], vol_shape[3], vol_shape[4]])
            if method == 'nn':
                vol_resize = tf.image.resize_nearest_neighbor(vol_reshape,
                                                              [vol_shape[2] * scale[0], vol_shape[3] * scale[1]])
            elif method == 'bilinear':
                vol_resize = tf.image.resize_bilinear(vol_reshape,
                                                      [vol_shape[2] * scale[0], vol_shape[3] * scale[1]])

            return tf.reshape(vol_resize,
                              [vol_shape[0], vol_shape[1], vol_shape[2] * scale[0], vol_shape[3] * scale[1],
                               vol_shape[4]])

    # Downsampling methods, including max-pooling, convolution with strides
    def downsampling_layer(self,
                           vol_in,
                           ksize=(3, 2, 2),
                           strides=(1, 2, 2),  # no downsampling along depth axis
                           padding='SAME',
                           downsampling_type='conv',
                           norm='IN',
                           name='downsampling'):
        if downsampling_type.upper() in ['MAXPOOL', 'MAX_POOL', 'MAXPOOLING', 'MAX_POOLING']:
            with tf.variable_scope(name):
                return tf.nn.max_pool3d(vol_in,
                                        ksize=[1, 1, ksize[1], ksize[2], 1],
                                        strides=[1, 1, strides[1], strides[2], 1],
                                        padding=padding)

        if downsampling_type.upper() in ['CONV', 'CONVOLUTION', 'STRIDED_CONV', 'STRIDED_CONVOLUTION']:
            return self.conv_layer(vol_in,
                                   channel_out=vol_in.shape[-1] * 2,
                                   name=name,
                                   kernel_size=ksize,
                                   strides=strides,
                                   norm=norm,
                                   init=tf.initializers.constant(np.prod(ksize)))  # init as average pooling

    # Upsampling methods, including convolution transpose, nearest neighbour/bilinear upsampling followed by convolution
    def upsampling_layer(self,
                         vol_in,
                         channel_out,
                         ksize=(3, 2, 2),
                         strides=(1, 2, 2),
                         upsampling_type='deconv',
                         norm='IN',
                         name='upsampling'):
        if not upsampling_type.upper() in ['DECONV', 'DECONVOLUTION', 'TRANSPOSED_CONVOLUTION', 'CONV_TRANSPOSE']:
            if upsampling_type.upper() in ['NN', 'NEAREST_NEIGHBOR', 'NEAREST_NEIGHBOUR']:
                vol_resized = self.resize_layer(vol_in, scale=strides[1:3], method='nn')
            elif upsampling_type.upper() in ['BILINEAR']:
                vol_resized = self.resize_layer(vol_in, scale=strides[1:3], method='bilinear')

            return self.conv_layer(vol_resized,
                                   channel_out,
                                   name=name,
                                   kernel_size=ksize,
                                   norm=norm)

        return self.deconv_layer(vol_in,
                                 channel_out,
                                 name=name,
                                 kernel_size=ksize,
                                 strides=strides,
                                 norm=norm)

    def activation_layer(self, x, activation_type):
        if activation_type.upper() == 'RELU':
            return tf.nn.relu(x)
        if activation_type.upper() == 'PRELU':
            alphas = tf.get_variable('alpha', x.get_shape()[-1],
                                     initializer=tf.constant_initializer(0.0),
                                     dtype=tf.float32)
            return tf.nn.relu(x) + tf.multiply(alphas, (x - tf.abs(x))) * 0.5
        if activation_type.upper() == 'ELU':
            return tf.nn.elu(x)

        return x

    # Normalization, default instance normalization
    def normalization(self, vol_in, norm_type, name, epsilon=1e-12):
        if norm_type.upper() in ['IN', 'INSTANCE_NORM', 'INSTANCE_NORMALIZATION']:
            return self.instance_normalization(vol_in, name, epsilon)
        if norm_type.upper() in ['LN', 'LAYER_NORM', 'LAYER_NORMALIZATION']:
            return self.layer_normalization(vol_in, name, epsilon)
        if norm_type.upper() in ['BN', 'BATCH_NORM', 'BATCH_NORMALIZATION']:
            return self.batch_normalization(vol_in, name, epsilon)

    def instance_normalization(self, vol_in, name, epsilon=1e-12):
        shape = vol_in.shape
        n_channels = shape[-1]
        mean, var = tf.nn.moments(vol_in, [1, 2, 3], keep_dims=True)
        with tf.variable_scope(name):
            gamma = tf.get_variable(name='gamma', shape=n_channels, dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable(name='beta', shape=n_channels, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))

            return tf.nn.batch_normalization(vol_in, mean, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    def layer_normalization(self, vol_in, name, epsilon=1e-12):
        shape = vol_in.shape
        n_channels = shape[-1]
        mean, var = tf.nn.moments(vol_in, [1, 2, 3, 4], keep_dims=True)
        with tf.variable_scope(name):
            gamma = tf.get_variable(name='gamma', shape=n_channels, dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable(name='beta', shape=n_channels, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))

        return tf.nn.batch_normalization(vol_in, mean, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    def batch_normalization(self, vol_in, name, epsilon=1e-12):
        shape = vol_in.shape
        n_channels = shape[-1]
        mean, var = tf.nn.moments(vol_in, [0, 1, 2, 3], keep_dims=True)
        with tf.variable_scope(name):
            gamma = tf.get_variable(name='gamma', shape=n_channels, dtype=tf.float32,
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable(name='beta', shape=n_channels, dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.0))

            return tf.nn.batch_normalization(vol_in, mean, var, offset=beta, scale=gamma, variance_epsilon=epsilon)

    def result(self, logits, labels, n_classes):
        foreground = tf.cast(
            tf.greater_equal(
                tf.slice(tf.nn.softmax(logits=logits),
                         begin=[0, 0, 0, 0, 1],
                         size=[-1, -1, -1, -1, n_classes - 1]),
                y=0.5),
            dtype=tf.int32)
        background = tf.subtract(x=1, y=foreground)
        classes = tf.concat(values=[background, foreground], axis=3)

        foreground_gt = tf.cast(x=tf.slice(input_=labels, begin=[0, 0, 0, 0, 1], size=[-1, -1, -1, -1, 1]),
                                dtype=tf.int32)
        intersection = tf.bitwise.bitwise_and(foreground, foreground_gt)
        union = tf.bitwise.bitwise_or(foreground, foreground_gt)

        iou_1 = tf.constant(1, dtype=tf.float32)
        iou_2 = tf.cast(tf.reduce_sum(intersection), dtype=tf.float32) / tf.cast(tf.reduce_sum(union), dtype=tf.float32)
        iou = tf.cond(tf.equal(tf.reduce_sum(union), 0), lambda: iou_1, lambda: iou_2)

        prediction = {"probabilities": tf.nn.softmax(logits=logits, name="probabilities"),
                      "masks": classes,
                      'IoU': iou,
                      'Or': tf.reduce_sum(union),
                      'And': tf.reduce_sum(intersection)
                      }

        return prediction
