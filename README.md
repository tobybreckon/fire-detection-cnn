# Experimentally Defined Convolutional Neural Network Architecture Variants for Non-temporal Real-time Fire Detection

Tested using Python 3.4.6, [TensorFlow 1.13.0](https://www.tensorflow.org/install/), [tflearn 0.3](http://tflearn.org/) and [OpenCV 3.3.1 / 4.0.x](http://www.opencv.org)

(requires opencv extra modules - ximgproc module for superpixel segmentation)

## Architectures:
![FireNet](https://github.com/tobybreckon/fire-detection-cnn/blob/master/images/FireNet.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FireNet architecture (above)
![InceptionV1-onFire](https://github.com/tobybreckon/fire-detection-cnn/blob/master/images/InceptionV1-OnFire.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;InceptionV1-OnFire architecture (above)

## Abstract:

_"In  this  work  we  investigate  the  automatic  detection  of  fire pixel  regions  in  video  (or  still)  imagery  within  real-time
bounds without reliance on temporal scene information.  As an extension to prior work in the field, we consider the performance  of  experimentally  defined,  reduced  complexity  deep convolutional neural network (CNN) architectures for this task. Contrary to contemporary trends in the field, our work illustrates
maximal accuracy of 0.93 for whole image binary fire detection (1),  with  0.89  accuracy  within  our  superpixel  localization
framework  can  be  achieved (2),  via  a  network  architecture  of significantly reduced complexity. These reduced architectures
additionally  offer  a  3-4  fold  increase  in  computational  performance offering up to 17 fps processing on contemporary
hardware  independent  of  temporal  information (1).    We  show the  relative  performance  achieved  against  prior  work  using
benchmark datasets to illustrate maximally robust real-time fire region detection."_

(1) using InceptionV1-OnFire CNN model (2) using SP-InceptionV1-OnFire CNN model

[[Dunnings and Breckon, In Proc. International Conference on Image Processing IEEE, 2018](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)]



---

## Reference implementation:
Our binary detection (FireNet / InceptionV1-OnFire) architectures determine whether an image frame contains fire globally, whereas the superpixel based approach breaks down the frame into segments and performs classification on each superpixel segment to provide in-frame localization.

This respository contains the ```firenet.py``` and ```inceptionV1OnFire.py``` files corresponding to the two binary (full-frame) detection models from the paper. In addition the ```superpixel-inceptionV1OnFire.py``` file corresponds to the superpixel based in-frame fire localization from the paper.

 To use these scripts the pre-trained network models must be downloaded using the shell script ```download-models.sh``` which will create an additional ```models``` directory containing the network weight data (on Linux/MacOS). Alternatively, you can manually download the pre-trained network models from [http://dx.doi.org/10.15128/r19880vq98m](http://dx.doi.org/10.15128/r19880vq98m) and unzip them to a directory called  ```models``` in the same place as the python files.

The superpixel based approach was trained to perform superpixel based fire detection and localization within a given frame as follows:
  * image frame is split into segments using SLIC superpixel segmentation technique.
  * the SP-InceptionV1-OnFire convolutional architecture, trained to detect fire in a given superpixel segment, is used on each superpixel.
  * at run-time, this SP-InceptionV1-OnFire, network is run on every superpixel from the SLIC segmentation output.

Training datasets:

* The custom dataset used for training and evaluation can be found on [Durham Collections](https://collections.durham.ac.uk/collections/r1ww72bb497) (together with the trained network models). **As a temporary measure (as of March 2019)**, please download the full dataset from https://durhamuniversity.box.com/s/bxp9bvfcz3anb3lk2f4f3z3rtqyug5b1 _(no login required, we have updated the record on Durham Collections to reflect this)_.

* In addition, standard datasets such as [furg-fire-dataset](https://github.com/steffensbola/furg-fire-dataset) were also used for training and evaluation.

![](https://github.com/tobybreckon/fire-detection-cnn/blob/master/images/slic-stages.png)
Original frame (left), Frame after superpixel segmentation (middle), Frame after superpixel fire prediction (right)

---
## Instructions to test pre-trained models:

To download and test the supplied code and pre-trained models (with TFLean/OpenCV installed) do:

```
$ git clone https://github.com/tobybreckon/fire-detection-cnn.git
$ cd fire-detection-cnn
$ sh ./download-models.sh
$ python firenet.py models/test.mp4
$ python inceptionV1OnFire.py models/test.mp4
$ python superpixel-inceptionV1OnFire.py models/test.mp4
```

---

## Instructions to use pre-trained models with other frameworks:

To convert the supplied pre-trained models from TFLearn checkpoint format to protocol buffer (.pb) format (used by [OpenCV](http://www.opencv.org) DNN, [TensorFlow](https://www.tensorflow.org/), ...) do:


```
$ cd converter
$ python firenet-to-protobuf.py
$ python inceptionV1OnFire-to-protobuf.py
```

This creates three ```.pb``` files inside the ```converter``` directory (```firenet.pb``` / ```inceptionv1onfire.pb```/```sp-inceptionv1onfire.pb```) which can then be tested with the  [OpenCV](http://www.opencv.org) DNN module (for example, using OpenCV > 4.1.0-pre) from within the same directory:

```
$ python test-pb-opencv.py
```

(N.B. for the superpixel network, the test script just checks loading and inference with the ```.pb``` loaed model but does not supply an actual superpixel image - just any test image, hence inference fails to detect the fire correctly for the example only).

**To convert to to other frameworks** (such as PyTorch, MXNet, Keras, ...) from this tensorflow format (protocol buffer, .pb): - please see the extensive deep neural network model conversion tools offered by the [MMdnn](https://github.com/Microsoft/MMdnn) project.

---

## Example video:
[![Examples](https://github.com/tobybreckon/fire-detection-cnn/blob/master/images/slic-ex.png)](https://youtu.be/RcNj8aMDer4)
Video Example - click image above to play.

---

## Reference:

If making use of this work in any way (including our [pretrained models](http://dx.doi.org/10.15128/r19880vq98m) or [dataset](http://dx.doi.org/10.15128/r2d217qp536)), please reference the following:

[Experimentally defined Convolutional Neural Network Architecture Variants for Non-temporal Real-time Fire Detection](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)
(Dunnings and Breckon), In Proc. International Conference on Image Processing IEEE, 2018.
```
@InProceedings{dunnings18fire,
  author =     {Dunnings, A. and Breckon, T.P.},
  title =      {Experimentally defined Convolutional Nerual Network Architecture Variants for Non-temporal Real-time Fire Detection},
  booktitle =  {Proc. International Conference on Image Processing},
  pages =      {1558-1562},
  year =       {2018},
  month =      {September},
  publisher =  {IEEE},
  doi = 	 {10.1109/ICIP.2018.8451657},
  keywords =   {simplified CNN, deep learning, fire detection, real-time, non-temporal, non-stationary visual fire detection},
}
```

### Acknowledgements:

Atharva (Art) Deshmukh (Durham University, _github and data set collation for publication_).

---
