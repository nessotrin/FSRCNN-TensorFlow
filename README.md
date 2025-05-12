# FSRCNN-TensorFlow
TensorFlow implementation of the Fast Super-Resolution Convolutional Neural Network (FSRCNN). This implements two models: FSRCNN which is more accurate but slower and FSRCNN-s which is faster but less accurate. Based on this [project](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html).

## This fork

Nothing that special about it. This is an optimized fork of [HelpSeeker's](https://github.com/HelpSeeker/FSRCNN-TensorFlow/) FSRCNNX-distortion. See [DISTORT.md](https://github.com/nessotrin/FSRCNN-TensorFlow/blob/master/DISTORT.md) for more info.

## Improvements
 * TensorFlow 2 (much faster training)
 * AdamW optimizer (thanks to TF2)
 * Lower RAM requirements
 * Ported improvements from [igv's](https://github.com/igv/FSRCNN-TensorFlow/) fork
 * Dataset caching
 * More realistic selectable compression types
 * Optimized shader generation
 * Large cleanups
 
## Runs 4x faster !

Much faster with comparable visual quality to [HelpSeeker's x2 16-0-4-1](https://github.com/HelpSeeker/FSRCNN-TensorFlow/releases/tag/1.2_distort). Optimized x2 12-0-4-1 models run at 4K60 on modern AMD iGPU laptops.
 

## Prerequisites
 * Python 3
 * TensorFlow >= 2
 * CUDA & cuDNN >= 6.0
 * Pillow
 * pillow_avif
 * FFmpeg
 * NumPy
 * Natsort

## Usage
You can specify epochs, learning rate, data directory, etc:
<br>
`python main.py --distort --epoch 1000 --batch_size 4096 --learning_rate 1e-4 --checkpoint_dir my_checkpoint --train_dir Train --test_dir Test --test_image test_image.png,test_image2.png`

Dump weights:
<br>
`python main.py --checkpoint_dir my_checkpoint --save-params my_cool_name`

Make an MPV compatible shader:
<br>
`python gen_v2.py params/weights_2_12_0_4_1.my_cool_name.txt`

Check `main.py` for all the possible flags.

## Result (Outdated)

Original butterfly image:

![orig](https://github.com/igv/FSRCNN-Tensorflow/blob/master/Test/Set5/butterfly_GT.bmp?raw=true)


Ewa_lanczos interpolated image:

![ewa_lanczos](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/ewa_lanczos.png?raw=true)


Super-resolved image:

![fsrcnn](https://github.com/igv/FSRCNN-Tensorflow/blob/master/result/fsrcnn.png?raw=true)

## Additional datasets

* [General-100](https://drive.google.com/open?id=0B7tU5Pj1dfCMVVdJelZqV0prWnM)

## TODO

* Release weights and shaders

## References

* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow)

* [liliumao/Tensorflow-srcnn](https://github.com/liliumao/Tensorflow-srcnn) 

* [carpedm20/DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow) 
