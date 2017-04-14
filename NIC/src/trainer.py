from helpers import Helpers
import logging
import os
import tensorflow as tf
import time

EPSILON = 1e-12
DISC_GRAD_CLIP = .01
DISC_PER_GEN = 1
ADAM_B1 = .5
ADAM_B2 = .9
PENALTY_WEIGHT = 10.


class Trainer:
    def __init__(self, session, gen, disc, opts):
        self.disc = disc
        self.gen = gen
        self.lib_dir = Helpers.get_lib_dir()
        self.session = session

        # Check if there are training examples available and config logging
        Helpers.check_for_examples()
        Helpers.config_logging()

        # Assign each option as self.option_title = value
        for key, value in opts.items():
            eval_string = "self.%s = %s" % (key, value)
            exec(eval_string)

    def train(self):
        # Set initial training shapes and placeholders
        x_shape = [None, self.training_height, self.training_width, 1]
        y_shape = x_shape[:3] + [2]

        x_ph = tf.placeholder(dtype=tf.float32, shape=x_shape, name='conditional_placeholder')
        y_ph = tf.placeholder(dtype=tf.float32, shape=y_shape, name='label_placeholder')

        # Build the generator to setup layers and variables
        self.gen.build(x_ph)

        # Generate a sample and attain the probability that the sample and the target are from the real distribution
        sample = self.gen.output

        prob_sample = self.disc.predict(sample, x_ph,)
        prob_target = self.disc.predict(y_ph, x_ph, reuse_scope=True)

        # Optimization ops for the generator
        gen_loss = -tf.reduce_mean(prob_sample)
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        gen_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=ADAM_B1, beta2=ADAM_B2)
        gen_grads = gen_opt.compute_gradients(gen_loss, gen_vars)
        # gen_grads = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gen_grads_]
        gen_update = gen_opt.apply_gradients(gen_grads)

        # Optimization ops for the discriminator
        disc_loss = tf.reduce_mean(prob_sample - prob_target)
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        disc_opt = tf.train.AdamOptimizer(self.learning_rate, beta1=ADAM_B1, beta2=ADAM_B2)
        #disc_grads = disc_opt.compute_gradients(disc_loss, disc_vars)
        penalty_grads = tf.gradients(prob_target, sample)
        penalty_grads_norm = tf.global_norm(penalty_grads)
        disc_loss_with_penalty = disc_loss + PENALTY_WEIGHT * tf.square(penalty_grads_norm - 1.)
        disc_grads = disc_opt.compute_gradients(disc_loss_with_penalty, disc_vars)
        disc_update = disc_opt.apply_gradients(disc_grads)

        # Training data retriever ops
        example = self.next_example(height=self.training_height, width=self.training_width)
        example_condition = tf.slice(example, [0, 0, 2], [self.training_height, self.training_width, 1])
        example_condition = tf.div(example_condition, 255.)
        example_label = tf.slice(example, [0, 0, 0], [self.training_height, self.training_width, 2])

        capacity = self.batch_size * 2
        batch_condition, batch_label = tf.train.batch([example_condition, example_label], self.batch_size,
                                                      num_threads=4,
                                                      capacity=capacity)

        # delete this when done (retrieves image to render while training)
        CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        filenames_ = tf.train.match_filenames_once(CURRENT_PATH + '/../lib/train/banana.jpg')
        filename_queue_ = tf.train.string_input_producer(filenames_)
        r = tf.WholeFileReader()
        fn_, f_ = r.read(filename_queue_)
        rgb_ = tf.image.decode_jpeg(f_, channels=3)
        rgb_ = tf.image.resize_images(rgb_, [self.training_height, self.training_width])
        img_ = tf.image.rgb_to_hsv(rgb_)
        img_ = tf.expand_dims(img_, axis=0)
        v_ = tf.slice(img_, [0, 0, 0, 2], [1, self.training_height, self.training_width, 1]) / 255.
        colored_sample = tf.image.hsv_to_rgb(tf.concat(axis=3, values=[sample, tf.multiply(v_, 255.)])) / 255.

        # Start session and begin threading
        logging.info("Initializing session and begin threading..")
        self.session.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        saver = tf.train.Saver(gen_vars)

        for i in range(self.iterations):
            try:
                # Update discriminator DISC_PER_GEN times
                for _ in range(DISC_PER_GEN):
                    feed_dict = {x_ph: batch_condition.eval(), y_ph: batch_label.eval()}
                    __, d_loss = self.session.run([disc_update, disc_loss], feed_dict=feed_dict)

                # Update generator
                feed_dict = {x_ph: batch_condition.eval()}
                _, g_loss = self.session.run([gen_update, gen_loss], feed_dict=feed_dict)

                # Print current epoch number and errors if warranted
                if i % self.print_every == 0:
                    total_loss = g_loss + d_loss
                    log1 = "Epoch %06d || Total Loss %.010f || " % (i, total_loss)
                    log2 = "Generator Loss %.010f || Discriminator Loss %.010f" % (g_loss, d_loss)
                    logging.info(log1 + log2)

                    # test out delete when done
                    self.disc.is_training = False
                    self.gen.is_training = False
                    rgb = self.session.run(colored_sample, feed_dict={x_ph: v_.eval()})
                    Helpers.render_img(rgb)
                    self.disc.is_training = True
                    self.gen.is_training = True

                # Save a checkpoint of the model
                if i % self.save_every == 0:
                    model_path = self.lib_dir + '/generator_%s' % time.time()
                    self.__save_model(saver, model_path)

            except tf.errors.OutOfRangeError:
                next

        # Alert that training has been completed and print the run time
        elapsed = time.time() - start_time
        logging.info("Training complete. The session took %.2f seconds to complete." % elapsed)
        coord.request_stop()
        coord.join(threads)

        # Save the trained model and close the tensorflow session
        model_path = self.lib_dir + '/generator_%s' % time.time()
        self.__save_model(saver, model_path)
        self.session.close()

    @staticmethod
    # Returns an image in both its grayscale and rgb formats
    def next_example(height, width):
        # Ops for getting training images, from retrieving the filenames to reading the data
        regex = Helpers.get_training_dir() + '/*.jpg'
        filenames = tf.train.match_filenames_once(regex)
        filename_queue = tf.train.string_input_producer(filenames)
        reader = tf.WholeFileReader()
        _, file = reader.read(filename_queue)

        img = tf.image.decode_jpeg(file, channels=3)
        img = tf.image.resize_images(img, [height, width])
        img = tf.image.rgb_to_hsv(img)
        return img

    def __save_model(self, saver, path):
        logging.info("Proceeding to save weights at '%s'" % path)
        saver.save(self.session, path)
        logging.info("Weights have been saved.")
