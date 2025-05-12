from utils import (
  multiprocess_train_setup,
  test_input_setup,
  save_params,
  merge
)

import time
import os
import importlib
from random import randrange

import numpy as np
import tensorflow as tf

from PIL import Image
import pdb
import sys

from pathlib import Path


# Based on http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html
class Model(object):
  
  def __init__(self, config):
    self.arch = config.arch
    self.fast = config.fast
    self.epoch = config.epoch
    self.scale = config.scale
    self.radius = config.radius
    self.batch_size = config.batch_size
    self.learning_rate = config.learning_rate
    self.distort = config.distort
    self.save_params = config.save_params
    self.test_image = config.test_image

    self.padding = 4
    # Different image/label sub-sizes for different scaling factors x2, x3, x4
    scale_factors = [[40 + self.padding, 40], [20 + self.padding, 40], [14 + self.padding, 42], [12 + self.padding, 48]]
    self.image_size, self.label_size = scale_factors[self.scale - 1]
    self.stride = self.image_size - self.padding - 1

    self.checkpoint_dir = config.checkpoint_dir
    self.train_dir = config.train_dir
    self.test_dir = config.test_dir
    self.rebuild_dataset = config.rebuild_dataset
    self.init_model()


  def init_model(self):
    model = importlib.import_module(self.arch)
    self.modelContainer = model.FSRCNNModel(self.scale)
    self.model = self.modelContainer.getModel()
    self.optimizer = tf.keras.optimizers.AdamW(self.learning_rate)
    self.global_step = tf.Variable(0, trainable=False, name="global_step")

    model_dir = "%s_%s_%s_%s" % (self.modelContainer.name.lower(), self.label_size, '-'.join(str(i) for i in self.modelContainer.getParams()[1:]), "r"+str(self.radius))
    self.model_dir = os.path.join(self.checkpoint_dir, model_dir)


    self.ckpt = tf.train.Checkpoint(step=self.global_step, 
                                            optimizer=self.optimizer,
                                            net=self.model)
    self.saver = tf.train.CheckpointManager(self.ckpt, self.model_dir, max_to_keep=None)


  def run(self):
    print(" [*] Reading checkpoints...")
    if self.load():
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed, starting from scratch...")


    if self.save_params is not None:
      save_params(self.model, self.modelContainer.getParams(), self.save_params)
    else:
      self.run_train()

  def run_tests(self,test_data,test_label):
    self.measure_test_metrics(test_data, test_label)
    if self.test_image:
      images = self.test_image.split(",")
      for image in images:
        self.run_on_image(Path(image),Path(self.checkpoint_dir) / Path(f"{Path(image).stem}.{self.ckpt.step.numpy()}.png"))

  @tf.function
  def train_step(self, lr, hr):
    with tf.GradientTape() as tape:
        sr = self.model(lr, training=True)
        loss = self.modelContainer.loss(hr, sr)
    gradient = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradient, self.model.trainable_variables))
    return loss


  def run_train(self):
    start_time = time.time()
    print("Beginning training setup...")
    train_data, train_label = multiprocess_train_setup(self)
    test_data, test_label = test_input_setup(self)
    print(f'Training setup took {time.time() - start_time} seconds')


    print("Training...")
    start_time = time.time()

    self.run_tests(test_data,test_label)

    for ep in range(self.epoch):
      # Run by batch images
      batch_idxs = len(train_data) // self.batch_size
      batch_average = 0
      for idx in range(0, batch_idxs):
        batch_images = train_data[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = train_label[idx * self.batch_size : (idx + 1) * self.batch_size]

        exp = randrange(3)
        if exp==0:
            images = batch_images
            labels = batch_labels
        elif exp==1:
            k = randrange(3)+1
            images = np.rot90(batch_images, k, (1,2))
            labels = np.rot90(batch_labels, k, (1,2))
        elif exp==2:
            k = randrange(2)
            images = batch_images[:,::-1] if k==0 else batch_images[:,:,::-1]
            labels = batch_labels[:,::-1] if k==0 else batch_labels[:,:,::-1]
        

        err_raw = self.train_step(images, labels)
        err = tf.reduce_mean(err_raw).numpy()
        self.ckpt.step.assign_add(1)

        batch_average += err

        counter = self.ckpt.step.numpy()
        if counter % (100) == 0:
          print("Epoch: [%4d], step: [%9d]/[%6d], time: [%11.4f], loss: [%.8f]" \
            % ((ep+1), counter, batch_idxs, time.time() - start_time, err))

        if counter % (10000) == 0:
          self.run_tests(test_data,test_label)
          self.save()

      batch_average = float(batch_average) / batch_idxs
      print("|================================================================================| => Epoch %4d loss: [%.8f]"%(ep+1,batch_average))


    print(f'Finished training. Ran for {time.time() - start_time} seconds.')

    self.run_tests(test_data,test_label)
    self.save()

    # Linux desktop notification when training has been completed
    # title = "Training complete - FSRCNN"
    # notification = "{}-{}-{} done training after {} epochs".format(self.image_size, self.label_size, self.stride, self.epoch);
    # notify_command = 'notify-send "{}" "{}"'.format(title, notification)
    # os.system(notify_command)

  def measure_test_metrics(self,test_data, test_label):
    img1_list = []
    img2_list = []

    start_time = time.time()
    print(f"Testing {len(test_data)} samples... ", end='', flush=True)
    batch_count = len(test_data) // self.batch_size
    for batch_id in range(batch_count):
      # print(data.shape,label.shape,test_data.shape,test_label.shape)
      # print(f"{batch_id}/{batch_count}")
      data = test_data[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]
      label = test_label[batch_id * self.batch_size : (batch_id + 1) * self.batch_size]
      result = self.model(lr)
      img1_list.append(label)
      img2_list.append(result)

    img1_list = np.concatenate(img1_list)
    img2_list = np.concatenate(img2_list)
    print("Computing PSNR.. ", end='', flush=True)
    test_psnr =  self.sess.run(tf.image.psnr(self.test_label, self.test_result, 1), feed_dict={self.test_label: img1_list, self.test_result:img2_list})
    print("SSIM.. ", end='', flush=True)
    test_ssim =  self.sess.run(tf.image.ssim(self.test_label, self.test_result, 1), feed_dict={self.test_label: img1_list, self.test_result:img2_list})
    print("MSSIM.. ", end='', flush=True)

    #run MSSIM as batches otherwise it crashes with large test datasets
    test_mssim =  []
    batch_size = min(16384,self.batch_size) #100000 crashes on Nvidia with a weird error
    num_batch = min(img1_list.shape[0] // batch_size,100) #go faster
    for b in range(num_batch):
      test_mssim.append(self.sess.run(tf.image.ssim_multiscale(self.test_label, self.test_result, 1,(0.2856/0.822, 0.3001/0.822, 0.2363/0.822),filter_size=9), feed_dict={self.test_label: img1_list[b*batch_size:b*batch_size+batch_size], self.test_result:img2_list[b*batch_size:b*batch_size+batch_size]}))
    test_mssim = np.concatenate(test_mssim)

    print("Tested %d samples in %.3f seconds." % (len(test_psnr), time.time() - start_time))

    print("[MIN]   PSNR: %9.6f, SSIM: %.6f, MSSIM: %.6f" % (np.min(np.array(test_psnr))          , np.min(np.array(test_ssim))          ,np.min(np.array(test_mssim))          ))
    print("[05pct] PSNR: %9.6f, SSIM: %.6f, MSSIM: %.6f" % (np.percentile(np.array(test_psnr),5) , np.percentile(np.array(test_ssim),5) ,np.percentile(np.array(test_mssim),5) ))
    print("[AVG]   PSNR: %9.6f, SSIM: %.6f, MSSIM: %.6f" % (np.average(np.array(test_psnr))      , np.average(np.array(test_ssim))      ,np.average(np.array(test_mssim))      ))
    print("[MED]   PSNR: %9.6f, SSIM: %.6f, MSSIM: %.6f" % (np.median(np.array(test_psnr))       , np.median(np.array(test_ssim))       ,np.median(np.array(test_mssim))       ))
    print("[95pct] PSNR: %9.6f, SSIM: %.6f, MSSIM: %.6f" % (np.percentile(np.array(test_psnr),95), np.percentile(np.array(test_ssim),95),np.percentile(np.array(test_mssim),95)))
    print("[MAX]   PSNR: %9.6f, SSIM: %.6f, MSSIM: %.6f" % (np.max(np.array(test_psnr))          , np.max(np.array(test_ssim))          ,np.max(np.array(test_mssim))          ))

  
  def run_on_image(self,path,output_path):
    og_image = Image.open(path).convert('YCbCr')
    Y_image,_,_ = og_image.split()#Y CB CR, keep Y
    (width,height) = og_image.size
    Y_array = np.frombuffer(Y_image.tobytes(), dtype=np.uint8).reshape((height, width))
    upscaled_Y_array = self.model(Y_array[np.newaxis,...,np.newaxis]/255).numpy()[0]*255
    result = merge(self, og_image, upscaled_Y_array)
    result.save(output_path,optimize=True)
    print(f"Upscaled {path} to {output_path}")


  def save(self):
    if not os.path.exists(self.model_dir):
        os.makedirs(self.model_dir)
        
    self.saver.save()

  def load(self):
    self.ckpt.restore(self.saver.latest_checkpoint)
