import net
import tensorflow as tf


class Discriminator(net.Net):
    def __init__(self, filter_size=4):
        net.Net.__init__(self)
        print("Initialized new 'Discriminator' instance")
        self.filter_size = filter_size
        self.stride1 = [1, 2, 2, 1]
        self.stride2 = [1, 1, 1, 1]
        self.is_training = True

    def predict(self, y, x, noise=None, reuse_scope=False):
        """
        Predicts the probability a given input belongs to a targeted sample distribution

        :param y: input tensor
        :param x: conditional tensor
        :param noise: regularizing gaussian noise tensor to add to xy
        :param reuse_scope:
        :return: probability tensor, logit tensor, average probability tensor
        """

        if noise is not None:
            y += noise

        xy = tf.concat(axis=3, values=[y, x])

        with tf.variable_scope('discriminator'):
            if reuse_scope:
                tf.get_variable_scope().reuse_variables()

            conv0 = self.conv_layer(xy, 64, act=self.leaky_relu, norm=False, pad='SAME', stride=self.stride1, name='conv0')

            conv1 = self.conv_layer(conv0, 128, act=self.leaky_relu, pad='SAME', stride=self.stride1, name='conv1')
            conv2 = self.conv_layer(conv1, 256, act=self.leaky_relu, pad='SAME', stride=self.stride1, name='conv2')
            conv3 = self.conv_layer(conv2, 512, act=self.leaky_relu, pad='SAME', stride=self.stride1, name='conv3')
            conv4 = self.conv_layer(conv3, 512, act=self.leaky_relu, pad='SAME', stride=self.stride1, name='conv4')

            conv5 = self.conv_layer(conv4, 512, act=self.leaky_relu, pad='SAME', stride=self.stride2, name='conv5')
            conv6 = self.conv_layer(conv5, 512, norm=False, pad='SAME', stride=self.stride2, name='conv6')

        return conv6

    # Batch normalize inputs to reduce covariate shift and improve the efficiency of training
    def batch_normalize(self, inputs, num_maps, decay=.9):
        with tf.variable_scope("batch_normalization"):
            # Trainable variables for scaling and offsetting our inputs
            scale = tf.get_variable('scale', initializer=tf.ones([num_maps], dtype=tf.float32), trainable=True)
            tf.summary.histogram('scale', scale)
            offset = tf.get_variable('offset', initializer=tf.constant(.1, shape=[num_maps]), trainable=True)
            tf.summary.histogram('offset', offset)

            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                with tf.name_scope(None):
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
