# *Neural-Image-Colorization* using adversarial networks

<img src="" width="640px" align="right">

This is a Tensorflow implementation of *[Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004v1.pdf)* that aims to infer a mapping from X to Y, where X is a single channel grayscale image and Y is 3-channel "colorized" version of that image. We make use of *[Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661.pdf)* conditioned on the input to teach a generative neural network the highly complex and abstract function of automatic photo colorization, a currently cutting-edge technology in the field of machine learning. 

The purpose of this repository is to port the image-to-image translation experiment over to TensorFlow.

#### Implementation Architecture

Architectural deficiencies presented in Berkley's paper were modified to improve training. Specifically, its logistic loss proved to be inefficient at learning the appropriate probability distribution for the generative model and in turn made the training process highly time consuming. To resolve this, architectural design elements were borrowed from the [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf). The final sigmoidal activation of the discriminator was removed which improved gradient flow and reduced [vanishing gradients](https://www.quora.com/What-is-the-vanishing-gradient-problem). Intuitively, this essentially makes the discriminator a critic; rather than discriminating between real and generated samples in a binary manner, it replies "how real" the given sample is and returns a number. 

Other notable elements borrowed from the Wasserstein GAN include using the [RMSProp](https://www.quora.com/What-is-an-intuitive-explanation-of-RMSProp) and proper clipping of the discriminator's gradients.

#### Image Representations

Rather than training a generative model to produce a novel three-channel RGB image, I take advantage of the [HSV](https://en.wikipedia.org/wiki/HSL_and_HSV) encoding to train it to produce only two channels. Because the <b>v</b>alue channel alone contains the grayscale information of an image, we only need to learn the appropriate <b>h</b>ue and <b>s</b>aturation channels, drastically reducing training run time; the generator does not need to generate the image from scratch, it only needs to learn the color hues and their respective intensities. 

## Results

<table style="width:100%">
  <tr>
    <th>Input</th> 
    <th>Output</th>
    <th>Ground-Truth</th>
  </tr>
  <tr>
    <td><img src="lib/readme_examples" width="100%"></td>
    <td><img src="lib/readme_examples" width=100%"></td> 
    <td><img src="lib/readme_examples" width=100%"></td> 
  </tr>
  <tr>
    <td><img src="lib/readme_examples" width="100%"></td>
    <td><img src="lib/readme_examples" width="100%"></td> 
    <td><img src="lib/readme_examples" width=100%"></td> 
  </tr>
  <tr>
    <td><img src="lib/readme_examples" width="100%"></td>
    <td><img src="lib/readme_examples" width="100%"></td> 
    <td><img src="lib/readme_examples" width=100%"></td> 
  </tr>
</table>

## Prerequisites

* [Python 3.5](https://www.python.org/downloads/release/python-350/)
* [TensorFlow](https://www.tensorflow.org/) (>= r1.0)
* [scikit-image](http://scikit-image.org/docs/dev/api/skimage.html)
* [NumPy](http://www.numpy.org/)

## Usage

To colorize a grayscale image using a trained model, invoke *colorize.py* and supply both the desired input image path and the saved model path. The input image does not need to be a single-channel image. Given inputs will automatically be downsmapled to one channel.

```sh
python colorize.py 'path/to/input/image' 'path/to/saved/model'
```

To train a generative model to colorize images invoke *train.py* after ensuring there is a training directory "train" in "/lib" containing a myriad of training examples. I suggest using a minimum of one million images. These images ough to be of the three-channel jpeg kind. Remember to check whether the images you have obtained are truly jpeg compressed. Often times creators of datasets will simply change the extensions of their images regardless of their types to ".jpg" without altering the actual data. 

```sh
python train.py 
```

## Files

* [colorize.py](./src/colorize.py)

    User script that colorizes a grayscale image given an already-trained model. 
    
* [generator.py](./src/generator.py)
    
    Contains the generative net that can colorize single-channel images when trained.
    
* [discriminator.py](./src/discriminator.py)
    
    Contains the discriminative net that can discriminate between synthesized colorized images and ground-truth images.
   
* [helpers.py](./src/helpers.py)
    
    Helper class containing various methods with their functions ranging from image retrieval to auxiliary math helpers.
    
* [net.py](./src/net.py)
    
    Contains the neural network super class with universal layer construction and instance normalization methods. 
    
* [train.py](./src/train.py)
    
    User script that trains a new generative model the can colorize any given grayscale image assuming the size and variety of the training set is sufficient.
   
* [trainer.py](./src/trainer.py)
    
    Contains a Trainer class that is responsible for training the generative adversarial networks and any related routines such as retrieving training data.