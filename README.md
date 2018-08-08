# Experimentally Defined Convolutional Neural Network Architecture Variants for Non-temporal Real-time Fire Detection

Tested using Python 3.4.6, [TensorFlow 1.9.0](https://www.tensorflow.org/install/), and [OpenCV 3.3.1](http://www.opencv.org)

(requires opencv extra modules - ximgproc module for superpixel segmentation)

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

_"In  this  work  we  investigate  the  automatic  detection  of  fire pixel  regions  in  video  (or  still)  imagery  within  real-time
bounds without reliance on temporal scene information.  As an extension to prior work in the field, we consider the performance  of  
experimentally  defined,  reduced  complexity  deep convolutional neural network architectures for this task. Contrary to contemporary trends in the field, our work illustrates
maximal accuracy of 0.93 for whole image binary fire detection,  with  0.89  accuracy  within  our  superpixel  localization
framework  can  be  achieved,  via  a  network  architecture  of significantly reduced complexity. These reduced architectures
additionally  offer  a  3-4  fold  increase  in  computational  performance offering up to 17 fps processing on contemporary
hardware  independent  of  temporal  information.    We  show the  relative  performance  achieved  against  prior  work  using
benchmark datasets to illustrate maximally robust real-time fire region detection."_

[[Dunnings and Breckon, In Proc. International Conference on Image Processing IEEE, 2018](http://breckon.eu/toby/publications/papers/dunnings18fire.pdf)]

---

## Reference implementation:
The binary detection approach shows whether frame contains fire globally, whereas the superpixel based approach breaks down the frame into segments and performs classification on each superpixel segment.

The two convolutional neural network architectures FireNet and InceptionV1-OnFire were based on existing architectures and were experimentally tweaked to achieve real-time performance on the global full-frame fire detection task. Full details are presented in the accompanying research paper.

The superpixel based approach was trained to perform superpixel based fire detection and localization within a given frame as follows:
  * The frame is split into segments using SLIC superpixel segmentation technique.
  * The SP-InceptionV1-OnFire convolutional architecture is trained to detect fire in a given superpixel segment.
  * At run-time, the SP-InceptionV1-OnFire network is run on every superpixel from the SLIC segmentation output.

Training datasets:

* The custom dataset used for training and evaluation can be found on [Durham Collections](https://collections.durham.ac.uk/collections/r1ww72bb497) together with the trained network models.
* Standard datasets such as [furg-fire-dataset](https://github.com/steffensbola/furg-fire-dataset) were also used for training and evaluation.

![](https://github.com/atharva333/fire-detection/blob/master/Images/slic-stages.png)
Original frame (left), Frame after superpixel segmentation (middle), Frame after superpixel fire prediction (right)

---
## Instructions to test pre-trained models:

```
$ git clone https://github.com/tobybreckon/fire-detection-cnn.git
$ cd fire-detection-cnn
$ sh ./download-models.sh
$ python firenet.py models/test.mp4
$ python inceptionV1-OnFire.py models/test.mp4
$ python superpixel-inceptionV1-OnFire.py models/test.mp4
```
---

* The main directory contains the ```firenet.py``` and ```inceptionV1-OnFire.py``` files corresponding to the two binary (full-frame) detection models from the paper.
* The additional ```superpixel-inceptionV1-OnFire.py``` file corresponds to the superpixel based in-frame fire localization from the paper.

* To use these scripts the pre-trained network models must be first downloaded using the shell script ```download-models.sh``` which will create a models directory to contain the network weight data.

---

## Example video:
[![Examples](https://github.com/atharva333/fire-detection/blob/master/Images/slic-ex.png)](https://youtu.be/RcNj8aMDer4)
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

### Acknowledgements:

Atharva (Art) Deshmukh (Durham University, _github and data set collation for publication_).

---
