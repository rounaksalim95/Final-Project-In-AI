import argparse
import generator
from helpers import Helpers
import os
import tensorflow as tf

# Arg parsing
parser = argparse.ArgumentParser(description='Colorize images using conditional generative adversarial networks.')
parser.add_argument('input', help='path to input image you would like to colorize (i.e., "~/Desktop/input.jpg")')
parser.add_argument('model', help='path to saved model (i.e., "lib/generator")')
args = parser.parse_args()
args.input = os.path.abspath(args.input)

with tf.Session() as sess:
    in_rgb, shape = Helpers.load_img(args.input)
    in_rgb = tf.convert_to_tensor(in_rgb, dtype=tf.float32)
    in_rgb = tf.expand_dims(in_rgb, axis=0)

    # Initialize new generative net and build its graph
    gen = generator.Generator()
    gen.build(in_rgb)
    sample = gen.output
    print ("finished 1 ...")

    # Ops for transforming a sample into a properly formatted rgb image
    img = tf.image.rgb_to_hsv(in_rgb)
    v = tf.slice(img, [0, 0, 0, 2], [1, shape[1], shape[2], 1]) / 255.
    colored_sample = tf.image.hsv_to_rgb(tf.concat(axis=3, values=[sample, tf.multiply(v, 255.)])) / 255.
    print ("finished 2 ...")

    # Initialize the TensorFlow session and restore the previously trained model
    sess.run(tf.global_variables_initializer())
    saved_path = args.model
    saver = tf.train.Saver()
    saver.restore(sess, saved_path)
    print ("finished 3 ...")

    # Generate colored sample and save it
    rgb = sess.run(colored_sample)
    Helpers.render_img(rgb)

    # FIN
    sess.close()
