# Experimentally Defined Convolutional Neural Network Architecture Variants for Non-temporal Real-time Fire Detection

Developed using Python 2.7.12, [TensorFlow 1.8.0](https://www.tensorflow.org/install/), and [OpenCV 3.3.1](http://www.opencv.org) (requires opencv extra modules - ximgproc module for superpixel segmentation),

## Architecture:
![FireNet](https://github.com/atharva333/fire-detection/blob/master/Images/FireNet.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FireNet architecture (above)
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
The binary detection approach shows whether frame contains fire, wheras the superpixel based approach breaks down the frame into segments and performs classification on each superpixel segment:

* The binary classifier was trained in order to achieve fire detection in a given frame globally.
  * The dataset used contained popular datasets and additional pictures taken from online fire videos.
  * The two convolutional neural network architectures FireNet and InceptionV1-OnFire were based on existing architectures and were experimentally tweaked to achieve real-time performance.
  
* The superpixel based approach was trained to perform superpixel based fire detection and localization within a given frame as follows:
  * The frame is split into segments using SLIC superpixel segmentation technique.
  * The classifier is trained on detecting fire given a superpixel segment.
  * The segments can then be used to show regions containing fire.
  * SP-InceptionV1-OnFire convolutional architecture was developed based on Inception module from the GoogLeNet architecture.
* The custom dataset used for training can be found on [Durham Collections](https://collections.durham.ac.uk/collections/r1ww72bb497)
* Standard datasets such as [furg-fire-dataset](https://github.com/steffensbola/furg-fire-dataset) were also used for training.

![](https://github.com/atharva333/fire-detection/blob/master/Images/slic-stages.png)
Vanilla frame (left), Frame after superpixel segmentation (middle), Frame after superpixel prediction (right)

---
## Instructions to test pretrained models:

```
$ git clone https://github.com/tobybreckon/fire-detection-cnn.git
$ cd fire-detection-cnn
$ chmod +x ./download-models.sh
$ ./download-models.sh
$ python binary.py models/test.mp4
```
---

* The main directory contains the ```binary.py``` and ```superpixel.py``` files
* To run the models you require a video file as argument - for example ```python binary.py test.mp4```
* The pretrained models will be downloaded using the shell script 'download-models.sh' which will create a models directory that contains the data
* The TensorFlow code for the FireNet and InceptionV1-OnFire are in the ```tflearn``` directory
* ```binary.py``` file can be run with both with FireNet and InceptionV1-OnFire, the model filepath should be chosen accordingly

---

## Example video:
[![Examples](https://github.com/atharva333/fire-detection/blob/master/Images/binary-ex.png)](https://youtu.be/RcNj8aMDer4)
Video Example - click image above to play.

---

## Reference:

[Experimentally defined Convolutional Neural Network Architecture Variants for Non-temporal Real-time Fire Detection](http://breckon.eu/toby/publications/papers/dunnings18fire.pdf)
(Dunnings and Breckon), In Proc. International Conference on Image Processing IEEE, 2018.
```
@InProceedings{dunnings18fire,
  author =     {Dunnings, A. and Breckon, T.P.},
  title =      {Experimentally defined Convolutional Nerual Network Architecture Variants for Non-temporal Real-time Fire Detection},
  booktitle =  {Proc. International Conference on Image Processing},
  pages =      {1-5},
  year =       {2018},
  month =      {September},
  publisher =  {IEEE},
  keywords =   {simplified CNN, fire detection, real-time, non-temporal, non-stationary visual fire detection},
}

```
---
