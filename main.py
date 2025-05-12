import os

from model import Model
import numpy as np
import tensorflow.compat.v1 as tf1

flags = tf1.app.flags
flags.DEFINE_string("arch", "FSRCNN", "Model name [FSRCNN]")
flags.DEFINE_integer("epoch", 100, "Number of epochs [10]")
flags.DEFINE_integer("batch_size", 4096, "The size of batch images [32]")
flags.DEFINE_float("learning_rate", 1e-4, "The learning rate of the adam optimizer [1e-4]")
flags.DEFINE_integer("scale", 2, "The size of scale factor for preprocessing input image [2]")
flags.DEFINE_integer("radius", 1, "Max radius of the deconvolution input tensor [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint (=project) directory [checkpoint]")
flags.DEFINE_string("train_dir", "Train", "Name of data directory to train on [Train]")
flags.DEFINE_string("test_dir", "Test", "Name of data directory to test on while training [Test]")
flags.DEFINE_boolean("distort", False, "Distort some images with AVIF compression artifacts after downscaling [False]")
flags.DEFINE_string("save_params", None, "Save weight and bias parameters with name [None]")
flags.DEFINE_boolean("rebuild_dataset", False, "Ignore cache and regenerate dataset [False]")
flags.DEFINE_string("test_image", None, "Path to an image (or comma separated list) that will be upscaled at regualar interval [None]")
FLAGS = flags.FLAGS


def main():
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  model = Model(config=FLAGS)
  model.run()
    
if __name__ == '__main__':
  main()
