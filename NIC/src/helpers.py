import logging
import numpy as np
import os
import tensorflow as tf
import time
import skimage
import skimage.io
import skimage.transform
from scipy.misc import toimage
from functools import reduce

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = os.path.abspath(DIR_PATH + '/../output/out_%.0f.jpg' % time.time())
CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
LIB_DIR = os.path.abspath(CURRENT_PATH + '/../lib/')
TRAINING_DIR = os.path.abspath(CURRENT_PATH + '/../lib/train/')
TRAINING_URL = 'http://msvocds.blob.core.windows.net/coco2014/train2014.zip'


class Helpers:
    def __init__(self):
        pass

    # Checks for training data to see if it's missing or not. Asks to download if missing.
    @staticmethod
    def check_for_examples():
        pass

    @staticmethod
    def config_logging():
        log_dir = DIR_PATH + '/../log/'
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            print('Directory "%s" was created for logging.' % log_dir)
        log_path = ''.join([log_dir, str(time.time()), '.log'])
        logging.basicConfig(filename=log_path, level=logging.INFO)

    @staticmethod
    def exit_program(rc=0, message="Exiting the program.."):
        logging.info(message)
        tf.get_default_session().close()
        exit(rc)

    @staticmethod
    def get_lib_dir():
        if not os.path.isdir(LIB_DIR):
            os.makedirs(LIB_DIR)
        return LIB_DIR

    @staticmethod
    def get_training_dir():
        return TRAINING_DIR

    # Returns a numpy array of an image specified by its path
    @staticmethod
    def load_img(path):
        # Load image [height, width, depth]
        img = skimage.io.imread(path) / 255.0
        assert (0 <= img).all() and (img <= 1.0).all()

        # Crop image from center
        short_edge = min(img.shape[:2])
        yy = int((img.shape[0] - short_edge) / 2)
        xx = int((img.shape[1] - short_edge) / 2)
        shape = list(img.shape)

        crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
        resized_img = skimage.transform.resize(crop_img, (shape[0], shape[1]))
        return resized_img, shape

    # Returns a resized numpy array of an image specified by its path
    @staticmethod
    def load_img_to(path, height=None, width=None):
        # Load image
        img = skimage.io.imread(path) / 255.0
        if height is not None and width is not None:
            ny = height
            nx = width
        elif height is not None:
            ny = height
            nx = img.shape[1] * ny / img.shape[0]
        elif width is not None:
            nx = width
            ny = img.shape[0] * nx / img.shape[1]
        else:
            ny = img.shape[0]
            nx = img.shape[1]

        if len(img.shape) < 3:
            img = np.dstack((img, img, img))

        return skimage.transform.resize(img, (ny, nx)), [ny, nx, 3]

    # Renders the generated image given a tensorflow session and a variable image (x)
    @staticmethod
    def render_img(img, display=False, path_out=None):
        if not path_out:
            path_out = os.path.abspath(DIR_PATH + '/../output/out_%.0f.jpg' % time.time())

        clipped_img = np.clip(img, 0., 1.)
        shaped_img = np.reshape(clipped_img, img.shape[1:])

        if display:
            toimage(shaped_img).show()

        if path_out:
            toimage(shaped_img).save(path_out)
