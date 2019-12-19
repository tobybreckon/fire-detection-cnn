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

_Which model should I use ?_ : for the best detection performance (i.e. true positive rate) and throughtput (speed, frames per second) use the FireNet model (example: ```firenet.py```); for a slighly lower false alarm rate (i.e. false positive rate, but only by 1%) but much lower throughtput (speed, frames per second) use the InceptionV1-OnFire model (example: ```inceptionV1OnFire.py```); for localization of the fire within the image use the superpixel InceptionV1-OnFire model (example: ```superpixel-inceptionV1OnFire.py```). For full details see paper - [[Dunnings and Breckon, 2018](https://breckon.org/toby/publications/papers/dunnings18fire.pdf)]

Training datasets:

* The custom dataset used for training and evaluation can be found on [Durham Collections](https://collections.durham.ac.uk/collections/r1ww72bb497) (together with the trained network models). A direct download link for the dataset is [https://collections.durham.ac.uk/downloads/r2d217qp536](https://collections.durham.ac.uk/downloads/r2d217qp536).
Datsaet DOI - [http://doi.org/10.15128/r2d217qp536](http://doi.org/10.15128/r2d217qp536). A download script ```download-dataset.sh``` is also provided which will create an additional ```dataset``` directory containing the training dataset (10.5Gb in size, works on Linux/MacOS).

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

To convert the supplied pre-trained models from TFLearn checkpoint format to protocol buffer (.pb) format (used by [OpenCV](http://www.opencv.org) DNN, [TensorFlow](https://www.tensorflow.org/), ...) and
also tflite (used with [TensorFlow](https://www.tensorflow.org/)) do:


```
$ cd converter
$ python firenet-conversion.py
$ python inceptionV1OnFire-conversion.py
```

This creates a set of six ```.pb``` and ```.tflite``` files inside the ```converter``` directory (```firenet.xxx``` / ```inceptionv1onfire.xxx```/```sp-inceptionv1onfire.xxx``` for ```xxx``` in ```[pb, tflite]```). These files can then be validated  with the [OpenCV](http://www.opencv.org) DNN module (OpenCV > 4.1.0-pre) and [TensorFlow](https://www.tensorflow.org/) against the original (tflearn) from within the same directory, as follows:

```
$ python firenet-validation.py
Load tflearn model from: ../models/FireNet ...OK
Load protocolbuf (pb) model from: firenet.pb ...OK
Load tflite model from: firenet.tflite ...OK
Load test video from ../models/test.mp4 ...
frame: 0        : TFLearn (original): [[9.999914e-01 8.576833e-06]]     : Tensorflow .pb (via opencv): [[9.999914e-01 8.576866e-06]]    : TFLite (via tensorflow): [[9.999914e-01 8.576899e-06]]: all equal test - PASS
frame: 1        : TFLearn (original): [[9.999924e-01 7.609045e-06]]     : Tensorflow .pb (via opencv): [[9.999924e-01 7.608987e-06]]    : TFLite (via tensorflow): [[9.999924e-01 7.608980e-06]]: all equal test - PASS
frame: 2        : TFLearn (original): [[9.999967e-01 3.373572e-06]]     : Tensorflow .pb (via opencv): [[9.999967e-01 3.373559e-06]]    : TFLite (via tensorflow): [[9.999967e-01 3.373456e-06]]: all equal test - PASS
frame: 3        : TFLearn (original): [[9.999968e-01 3.165212e-06]]     : Tensorflow .pb (via opencv): [[9.999968e-01 3.165221e-06]]    : TFLite (via tensorflow): [[9.999968e-01 3.165176e-06]]: all equal test - PASS
...
```

This can be similarly repeated with the ```inceptionV1OnFire-validation.py``` and ```sp-inceptionV1OnFire-validation.py``` validation scripts (N.B. here the superpixel inceptionV1OnFire network is being validated against the whole image frame rather than superpixels just for simply showing consistent output between the original and converted models).

**To convert to to other frameworks** (such as PyTorch, MXNet, Keras, ...) from these tensorflow formats: - please see the extensive deep neural network model conversion tools offered by the [MMdnn](https://github.com/Microsoft/MMdnn) project.

---

## Example video:
[![Examples](https://github.com/tobybreckon/fire-detection-cnn/blob/master/images/slic-ex.png)](https://youtu.be/RcNj8aMDer4)
Video Example - click image above to play.

---

## Reference:

If making use of this work in any way (including our [pretrained models](http://dx.doi.org/10.15128/r19880vq98m) or [dataset](http://dx.doi.org/10.15128/r2d217qp536)), _you must_ reference the following article in any report, publication, presentation, software release
or any other materials:

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

In addition the terms of the [LICENSE](LICENSE) must be adhered to.

### Acknowledgements:

Atharva (Art) Deshmukh (Durham University, _github and data set collation for publication_).

---
