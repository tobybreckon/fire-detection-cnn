# Experimentally Defined Convolutional Neural Network Architecture Variants for Non-temporal Real-time Fire Detection

Developed using Python 2.7.12, [TensorFlow 1.8.0](https://www.tensorflow.org/install/), and [OpenCV 3.3.1](http://www.opencv.org),

## Architecture:
![FireNet](https://github.com/atharva333/fire-detection/blob/master/Images/FireNet.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FireNet architecture (above)
![InceptionV1-onFire](https://github.com/atharva333/fire-detection/blob/master/Images/InceptionV1-OnFire.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;InceptionV1-OnFire architecture (above)

## Abstract:

_"In  this  work  we  investigate  the  automatic  detection  of  fire
pixel  regions  in  video  (or  still)  imagery  within  real-time
bounds without reliance on temporal scene information.  As
an extension to prior work in the field, we consider the perfor-
mance  of  experimentally  defined,  reduced  complexity  deep
convolutional neural network architectures for this task. Con-
trary to contemporary trends in the field, our work illustrates
maximal accuracy of 0.93 for whole image binary fire detec-
tion,  with  0.89  accuracy  within  our  superpixel  localization
framework  can  be  achieved,  via  a  network  architecture  of
signficantly reduced complexity. These reduced architectures
additionally  offer  a  3-4  fold  increase  in  computational  per-
formance offering up to 17 fps processing on contemporary
hardware  independent  of  temporal  information.    We  show
the  relative  performance  achieved  against  prior  work  using
benchmark datasets to illustrate maximally robust real-time
fire region detection."_

[[Dunnings and Breckon, In Proc. International Conference on Image Processing IEEE, 2018](http://breckon.eu/toby/publications/papers/dunnings18fire.pdf)]

---

## Reference implementation:
Produces a depth map output image based on a monocular color image input.
* The input RGB image will first be transformed into the style of the images captured from a highly realistic synthetic virtual environment, on which the depth prediction network is trained.
* The provided color image is used as the input to [CycleGAN](https://junyanz.github.io/CycleGAN/), which transforms the style of the image. Image style transfer is used as a method of domain adaptation.
* The style transferred image is used as the input to a model trained on synthetic images and can produce pixel-perfect depth outputs.
* The code provides an inference pipeline and can be run using the test harness: run_test.py
* Example images are provided in the 'Examples' directory.
* The training was in part performed based on the code from [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and we would like to thank the authors and contributors.


![](https://github.com/atapour/styleDepth-Inference/blob/master/imgs/sample.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Example of the results of the approach

---
## Instructions to test pretrained models:

```
$ git clone https://github.com/atharva333/fire-detection.git
$ cd fire-detection
$ chmod +x ./download-models.sh
$ ./download-models.sh
$ python binary.py models/test.mp4
```
---

* The main directory contains the binary.py and superpixel.py files
* To run the models you require a video file as argument - for example 'python binary.py test.mp4'
* The pretrained models will be downloaded using the shell script 'download-models.sh' which will create a models directory that contains the data
* The TensorFlow code for the FireNet and InceptionV1-OnFire are in the Code/Application/tflearn directory
* binary.py file can be run with both with FireNet and InceptionV1-OnFire, the model filepath should be chosen respectively

---


## Example video:
[![Examples](https://github.com/atharva333/fire-detection/blob/master/Images/binary-ex.png)] (https://vimeo.com/260393753)
Video Example - click image above to play.

---

## Reference:

[Experimentally defined Convolutional Neural Network Architecture Variants for Non-temporal Real-time Fire Detection](http://breckon.eu/toby/publications/papers/dunnings18fire.pdf)
(Dunnings and Breckon), In Proc. International Conference on Image Processing IEEE, 2018.
```
@InProceedings{dunnings18fire,
  author = 		{Dunnings, A. and Breckon, T.P.},
  title = 		{Experimentally defined Convolutional Nerual Network Architecture Variants for Non-temporal Real-time Fire Detection},
  booktitle = 	{Proc. International Conference on Image Processing},
  pages =		{1-5},
  year = 		{2018},
  month = 	 {September},
  publisher = 	{IEEE},
  keywords = 		{simplified CNN, fire detection, real-time, non-temporal, non-stationary visual fire detetction},
}

```
---
