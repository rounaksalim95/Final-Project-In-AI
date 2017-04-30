# Final-Project-In-AI

## Part 1. Using CNN to classify fruits
After trying decision tree and Nielsen Networks for classifying fruits, we used a basic TensorFlow Convolutional Neural Netowrk (CNN) model to test on the classfication problem.

To train the CNN model, please run 
```sh
python simple_cnn.py
```
The loss at each iteration will be printed (many hyperparameters can be modified, and this file is just for demo purposes).

Note: It took us a while to find out the best way to feed the data in our TensorFlow model. All temporary training images are stored in the 'data' folder. Specifically, images corresponding to different fruits are stored in different sub directory. We are using a python script `build_image_data.py` from the official TensorFlow website to convert this directory into two packaged files: `train-00000-of-00001` and `validation-00000-of-00001`. These two files are corresponding to training and testing data sets being used in the `simple_cnn.py`.

## Part 2. Neural Image Colorization
The project we adapted from [mohamedkeid](https://github.com/mohamedkeid/Neural-Image-Colorization) is stored in the `NIC` folder.

To train the GAN model, please run
```sh
python3 train.py 
```
However, this is also for demo purposes, since a typical training process would take at least 15 to 20 hours, and would generate 500Mb to 2Gb of temporary files.

Please note that this script requires several python modules:
* [Python 3.5](https://www.python.org/downloads/release/python-350/)
* [TensorFlow](https://www.tensorflow.org/) (>= r1.0)
* [scikit-image](http://scikit-image.org/docs/dev/api/skimage.html)
* [NumPy](http://www.numpy.org/)
