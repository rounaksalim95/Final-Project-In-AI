import argparse
import discriminator
import generator
import tensorflow as tf
from trainer import Trainer

# Model hyperparamaters
opts = {
    'batch_size': 4,
    'iterations': 1000,
    'learning_rate': .0002,
    'print_every': 10,
    'save_every': 50,
    'training_height': 256,
    'training_width': 256,
}


def parse_args():
    """
    Creates command line arguments with the same name and default values as those in the global opts variable
    Then updates opts using their respective argument values
    """

    # Parse command line arguments to assign to the global opt variable
    parser = argparse.ArgumentParser(description='colorize images using conditional generative adversarial networks')
    for opt_name, value in opts.items():
        parser.add_argument("--%s" % opt_name, default=value)

    # Update global opts variable using flag values
    args = parser.parse_args()
    for opt_name, _ in opts.items():
        opts[opt_name] = getattr(args, opt_name)

parse_args()
with tf.Session() as sess:
    # Initialize networks
    gen = generator.Generator()
    disc = discriminator.Discriminator()

    # Train them
    t = Trainer(sess, gen, disc, opts)
    t.train()
