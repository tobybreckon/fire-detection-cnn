# EXPERIMENTALLY DEFINED CONVOLUTIONAL NEURAL NETWORK ARCHITECTURE VARIANTS FOR NON-TEMPORAL REAL-TIME FIRE DETECTION

Developed using Python 2.7, [TensorFlow 1.8.0](https://www.tensorflow.org/install/), and [OpenCV 3.3.1](http://www.opencv.org),

![Examples](https://github.com/atharva333/fire-detection/images/SP_1.png)
[](https://github.com/atharva333/fire-detection/images/SP_Partial.png)
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

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
## Instructions to run the inference code using PyTorch 0.3.1:

```
$ git clone https://github.com/atharva333/fire-detection.git
$ cd fire-detection
$ chmod +x ./download-models.sh
$ ./download-models.sh
$ python binary.py models/test.mp4
```
---

The output results are written in a directory taken as an argument to the test harness ('./results' by default):
* the script entitled "download_pretrained_models.sh" will download the required pre-trained models and checks the downloaded file integrity using MD5 checksum.
* the checkpoints that are available for direct download were created using pyTorch 0.3.1 and will not work if you are using pyTorch 0.4.0. The provided python script named ' remove_running_stats.py' will remedy the situation.
* the file with the suffix "_original" is the original input image.
* the file with the suffix "_restyled" is the style transferred image.
* the file with the suffix "_depth" is the output depth image.

---


## Example:
[![Video Example](https://github.com/atapour/styleDepth-Inference/blob/master/imgs/thumbnail.jpg)](https://vimeo.com/260393753 "Video Example - Click to Play")

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Video Example - click image above to play.

---

## Reference:

[Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation via Image Style Transfer](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)
(A. Atapour-Abarghouei, T.P. Breckon), In Proc. Conf. Computer Vision and Pattern Recognition, 2018. [[pdf](http://breckon.eu/toby/publications/papers/abarghouei18monocular.pdf)] [[demo](https://vimeo.com/260393753)]

```
@InProceedings{abarghouei18monocular,
  author = 		{Atapour-Abarghouei, A. and Breckon, T.P.},
  title = 		{Real-Time Monocular Depth Estimation using Synthetic Data with Domain Adaptation},
  booktitle = 	{Proc. Computer Vision and Pattern Recognition},
  pages =		{1-8},
  year = 		{2018},
  month = 		{June},
  publisher = 	{IEEE},
  keywords = 		{monocular depth, generative adversarial network, GAN, depth map, disparity, depth from single image},
}

```
---
