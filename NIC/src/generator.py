import net
import tensorflow as tf


class Generator(net.Net):
    def __init__(self):
        net.Net.__init__(self)
        print("Initialized new 'Generator' instance")

    def build(self, x, nf=64, oc=2):
        """

        :param z: gaussian noise tensor
        :param x: conditional tensor
        :param nf: number of output filters
        :param oc: number of output channels
        :return:
        """
        with tf.variable_scope('generator'):
            # Encoder
            self.conv1e = self.conv_layer(x, nf, act=None, norm=False, name='conv1e')
            self.conv2e = self.conv_layer(self.conv1e, nf * 2, act=self.leaky_relu, name='conv2e')
            self.conv3e = self.conv_layer(self.conv2e, nf * 4, act=self.leaky_relu, name='conv3e')
            self.conv4e = self.conv_layer(self.conv3e, nf * 8, act=self.leaky_relu, name='conv4e')
            self.conv5e = self.conv_layer(self.conv4e, nf * 8, act=self.leaky_relu, name='conv5e')
            self.conv6e = self.conv_layer(self.conv5e, nf * 8, act=self.leaky_relu, name='conv6e')
            self.conv7e = self.conv_layer(self.conv6e, nf * 8, act=self.leaky_relu, name='conv7e')
            self.conv8e = self.conv_layer(self.conv7e, nf * 8, act=self.leaky_relu, name='conv8e')

            # U-Net decoder
            self.conv1d = self.__residual_layer(self.conv8e, self.conv7e, nf * 8, act=self.leaky_relu, drop=True, name='conv1d')
            self.conv2d = self.__residual_layer(self.conv1d, self.conv6e, nf * 8, act=self.leaky_relu, drop=True, name='conv2d')
            self.conv3d = self.__residual_layer(self.conv2d, self.conv5e, nf * 8, act=self.leaky_relu, drop=True, name='conv3d')
            self.conv4d = self.__residual_layer(self.conv3d, self.conv4e, nf * 8, act=self.leaky_relu, name='conv4d')
            self.conv5d = self.__residual_layer(self.conv4d, self.conv3e, nf * 4, act=self.leaky_relu, name='conv5d')
            self.conv6d = self.__residual_layer(self.conv5d, self.conv2e, nf * 2, act=self.leaky_relu, name='conv6d')
            self.conv7d = self.__residual_layer(self.conv6d, self.conv1e, nf, act=self.leaky_relu, name='conv7d')
            self.conv8d = self.__upsample_layer(self.conv7d, oc, norm=False, name='conv8d')
            self.output = tf.nn.tanh(self.conv8d, name='output')

    def __upsample_layer(self, inputs, out_size, name, act=tf.nn.relu, norm=True, drop=False):
        with tf.variable_scope(name):
            in_size = inputs.get_shape().as_list()[3]
            filters_shape = self.filter_shape + [out_size] + [in_size]
            filters = tf.get_variable('weights', initializer=tf.truncated_normal(filters_shape, stddev=.02))
            tf.summary.histogram('weights', filters)

            # Get dimensions to use for the deconvolution operator
            shape = tf.shape(inputs)
            out_height = shape[1] * self.sample_level
            out_width = shape[2] * self.sample_level
            out_size = filters_shape[2]
            out_shape = tf.stack([shape[0], out_height, out_width, out_size])

            # Deconvolve and normalize the biased outputs
            conv_ = tf.nn.conv2d_transpose(inputs, filters, output_shape=out_shape, strides=self.stride)
            bias = tf.get_variable('bias', initializer=tf.constant(.0, shape=[out_size]))
            tf.summary.histogram('bias', bias)
            conv = tf.nn.bias_add(conv_, bias)

            # Training related ops
            conv = self.batch_normalize(conv, self.is_training) if norm else conv
            conv = tf.nn.dropout(conv, keep_prob=self.dropout_keep) if drop else conv
            activations = act(conv)
            return activations

    def __residual_layer(self, inputs, skip, out_size, name, act=tf.nn.relu, norm=True, drop=False):
        """
        Upsamples a given input tensor and concatenates another given tensor to the output

        :param inputs:
        :param skip:
        :param out_size:
        :param name:
        :param act:
        :param norm:
        :param drop:
        :return:
        """

        conv = self.__upsample_layer(inputs, out_size, name, act=act, norm=norm, drop=drop)
        join = conv + skip
        return join

    # Batch normalize inputs to reduce covariate shift and improve the efficiency of training
    def batch_normalize(self, inputs, num_maps, decay=.9):
        with tf.variable_scope("batch_normalization"):
            # Trainable variables for scaling and offsetting our inputs
            scale = tf.get_variable('scale', initializer=tf.ones([num_maps], dtype=tf.float32), trainable=True)
            tf.summary.histogram('scale', scale)
            offset = tf.get_variable('offset', initializer=tf.constant(.1, shape=[num_maps]), trainable=True)
            tf.summary.histogram('offset', offset)

            # Mean and variances related to our current batch
            batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])

            # Create an optimizer to maintain a 'moving average'
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def ema_retrieve():
                return ema.average(batch_mean), ema.average(batch_var)

            # If the net is being trained, update the average every training step
            def ema_update():
                ema_apply = ema.apply([batch_mean, batch_var])

                # Make sure to compute the new means and variances prior to returning their values
                with tf.control_dependencies([ema_apply]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # Retrieve the means and variances and apply the BN transformation
            mean, var = tf.cond(tf.equal(self.is_training, True), ema_update, ema_retrieve)
            bn_inputs = tf.nn.batch_normalization(inputs, mean, var, offset, scale, self.epsilon)

        return bn_inputs
