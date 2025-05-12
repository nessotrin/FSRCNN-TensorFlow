"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
from math import ceil
import subprocess
import io
import random
import numpy as np
import multiprocessing
import time

from pathlib import Path
import tensorflow.compat.v1 as tf
from PIL import Image
import pillow_avif


FLAGS = tf.app.flags.FLAGS

compression_type = "h264" #h264 or avif
compression_debug = False #save distortion intermediate in "comp" folder

avif_qual_min = 60 #Rough equivalent to JPEG 50
avif_qual_max = 80 #Rough equivalent to JPEG 75

# h264_qual_min = 19 #Low distortion removal
# h264_qual_max = 22 #Low distortion removal
h264_qual_min = 22 #High distortion removal
h264_qual_max = 30 #High distortion removal

distortion_ratio=10 # 1/x chance of being NOT distorted

#Filenames and paths will be matched to these
keywords_of_renders = [] #Example: "my_game_render_directory","a_game_name"
keywords_of_renders_sample_lightly = [] #Only 1/5 images will be used. For datasets with very similar game screenshots.


def is_render(path):
  return any(filter(lambda x: x in path, keywords_of_renders))

def render_has_duplicates(path):
  return any(filter(lambda x: x in path, keywords_of_renders_sample_lightly))


def add_noise(img, min_noise=0.01, max_noise=0.1):
    width , height = img.size
    pix_count = width*height

    number_of_pixels = random.randint(int(pix_count*min_noise), int(pix_count*max_noise))
    arr = np.array(img)
    for i in range(number_of_pixels):
        x_coord=random.randint(0, width-1)
        y_coord=random.randint(0, height-1)
        color = random.randint(0, 255)
        arr[y_coord][x_coord][0] = color
        arr[y_coord][x_coord][1] = color
        arr[y_coord][x_coord][2] = color
    img = Image.fromarray(arr)

    return img

#https://stackoverflow.com/questions/67296517/is-it-possible-to-apply-h-264-compression-to-image
def compress_image_h264(image, quality):
    buf = io.BytesIO()
    cnt = 0

    width, height = image.size
    rot_x = random.randint(0, width)
    rot_y = random.randint(0, height)

    multiplier = random.uniform(-2, 2)
    rot_y = random.randint(0, height)

    #create mini video to simulate more than I-frames
    for i in reversed(range (1,50)):
        rot = image.copy().rotate(angle=i*multiplier,center=(rot_x,rot_y))
        rot = add_noise(rot)
        rot.save(buf, "PNG", optimize=False, compress_level=0)
        cnt += 1
    image.save(buf, "PNG", optimize=False, compress_level=0)
    buf.seek(0)

    # Use ffmpeg to compress the image using H.264 codec and MKV container
    ffmpeg_command = [
        'ffmpeg',
        '-y',                        # Overwrite output files without asking
        '-i', 'pipe:0',              # Input from stdin
        '-vcodec', 'libx264',        # Use H.264 codec
        '-preset', 'veryfast',       # Preset
        '-crf', str(quality),        # Quality parameter
        '-pix_fmt', 'yuv420p',       # Pixel format
        '-f', 'matroska',            # Use MKV container
        'pipe:1'                     # Output to stdout
    ]

    result = subprocess.run(
        ffmpeg_command,
        input=buf.read(),    # Pass PNG data to stdin
        stdout=subprocess.PIPE,      # Capture stdout
        stderr=subprocess.PIPE       # Capture stderr for debugging
    )

    if result.returncode != 0:
        print("FFmpeg error during compression:", result.stderr.decode())
        raise RuntimeError("FFmpeg compression failed")

    return cnt, result.stdout

def decompress_image_h264(compressed_data, width, height, cnt):
    # Use ffmpeg to decompress the image from H.264 to raw format
    ffmpeg_command = [
        'ffmpeg',
        '-i', 'pipe:0',              # Input from stdin
        '-f', 'rawvideo',            # Output raw video format
        '-pix_fmt', 'bgr24',         # Pixel format
        'pipe:1'                     # Output to stdout
    ]

    result = subprocess.run(
        ffmpeg_command,
        input=compressed_data,       # Pass compressed data to stdin
        stdout=subprocess.PIPE,      # Capture stdout
        stderr=subprocess.PIPE       # Capture stderr for debugging
    )

    if result.returncode != 0:
        print("FFmpeg error during decompression:", result.stderr.decode())
        raise RuntimeError("FFmpeg decompression failed")

    # Get the raw image data from stdout
    raw_image_data = result.stdout

    # Ensure we have enough data to reshape into the desired format
    expected_size = (cnt+1)*width * height * 3
    if len(raw_image_data) != expected_size:
        print("Unexpected raw image data size:", len(raw_image_data))
        raise ValueError(f"Cannot reshape array of size {len(raw_image_data)} into shape ({height},{width},3)")

    # Convert the raw data to a numpy array
    frame = np.flip(np.frombuffer(raw_image_data, dtype=np.uint8).reshape(((cnt+1),height, width, 3))[cnt],-1)
    return Image.fromarray(frame.astype('uint8'), 'RGB')

def apply_h264_compression(image,quality):
  frame_position, frame = compress_image_h264(image,quality)
  if compression_debug:
    with open(f"comp/test_{random.randint(1,100)}.h264.mkv", "wb") as outfile:
        outfile.write(frame)
  return decompress_image_h264(frame,*image.size, frame_position)

def preprocess(shared_dict,shared_dict_lock, path, scale, distort):

  """
  Preprocess single image file
    (1) Read original image
    (2) Converts to greyscale
    (3) Downscale by scale
    (4) Compress to introduce artifacts
  """
  print(f'Preprocessing "{path}"')

  shared_dict_lock.acquire()
  shared_dict["preprocessed"] += 1
  print("Preprocessed :",shared_dict["preprocessed"])
  num = shared_dict["preprocessed"]
  shared_dict_lock.release()
  
  try:
      og_image = Image.open(path)
      (og_width, og_height) = og_image.size

      image = og_image.crop((0,0,og_width-og_width%(2*scale), og_height-og_height%(2*scale)))
      (width, height) = image.size
  except Exception as e:
      print(f"===!! Failure to load image {path} !!===")
      raise e


  shared_dict_lock.acquire()
  if is_render(path):
    shared_dict["surf_render"] += width*height
  else:
    shared_dict["surf_photo"] += width*height
  shared_dict_lock.release()


  (new_width, new_height) = width // scale, height // scale
  scaled_image = image.resize((new_width, new_height), Image.LANCZOS)

  if distort==True and random.randrange(distortion_ratio):
      print("Distorting image.")

      og_scaled_image = scaled_image.copy()
      if compression_type == "avif":
        buf = io.BytesIO()
        quality = random.randrange(avif_qual_min, avif_qual_max+1, 5)
        scaled_image.convert('RGB').save(buf, "AVIF", quality=quality)
        buf.seek(0)
        scaled_image = Image.open(buf)
      elif compression_type == "h264":
        quality=random.randrange(h264_qual_min, h264_qual_max+1, 1)
        scaled_image = apply_h264_compression(scaled_image.convert('RGB'), quality) 
      else:
        print("Unsupported compression type.")
        os.exit(1)

      #save compression before/after for test purposes
      if compression_debug:
        both = Image.new('RGB', (og_scaled_image.size[0]*2,og_scaled_image.size[1]))
        both.paste(og_scaled_image)
        both.paste(scaled_image, (og_scaled_image.size[0],0))
        both.save(f"comp/test_{num}_q{quality}.png")
  else:
      print("Not distorting image.")

  input_ = np.frombuffer(scaled_image.convert('L').tobytes(), dtype=np.uint8).reshape((new_height, new_width))
  label_ = np.frombuffer(image.convert('L').tobytes(), dtype=np.uint8).reshape((height, width))

  return input_ / 255, label_ / 255

def prepare_data(dataset):
  """
  Args:
    dataset: choose train dataset or test dataset
    
    For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
  """
  from natsort import natsorted
  data_dir = os.path.join(os.getcwd(), dataset)
  data = []
  for files in ('**/*.bmp', '**/*.png'):
      data.extend(glob.glob(os.path.join(data_dir, files),recursive=True))
  data = natsorted(data)
  filtered_data = []
  real_c = 0
  render_c = 0
  for counter,d in enumerate(data):
    if is_render(d):
      if counter%5 != 0 and render_has_duplicates(d):
        continue
      render_c += 1
    else:
      real_c += 1      
    filtered_data.append(d)
  data = filtered_data
  random.shuffle(data)
  print(f"File list is composed of {real_c} photos and {render_c} renders.")

  return data

def modcrop(image, scale=3):
  """
  To scale down and up the original image, first thing to do is to have no remainder while scaling operation.
  
  We need to find modulo of height (and width) and scale factor.
  Then, subtract the modulo from height (and width) of original image size.
  There would be no remainder even after scaling operation.
  """
  if len(image.shape) == 3:
    h, w, _ = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w, :]
  else:
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
  return image

def train_input_worker(args):
  shared_dict,shared_dict_lock,image_data, config = args
  return preprocess_and_cut(shared_dict,shared_dict_lock,image_data, config)

def preprocess_and_cut(shared_dict,shared_dict_lock,image_data, config):
  image_size, label_size, stride, scale, padding, distort = config

  single_input_sequence, single_label_sequence = [], []


  input_, label_ = preprocess(shared_dict,shared_dict_lock,image_data, scale, distort)

  if len(input_.shape) == 3:
    h, w, _ = input_.shape
  else:
    h, w = input_.shape

  for x in range(0, h - image_size + 1, stride):
    for y in range(0, w - image_size + 1, stride):
      sub_input = input_[x : x + image_size, y : y + image_size]
      x_loc, y_loc = x + padding, y + padding
      sub_label = label_[x_loc * scale : x_loc * scale + label_size, y_loc * scale : y_loc * scale + label_size]

      sub_input = sub_input.reshape([image_size, image_size, 1])
      sub_label = sub_label.reshape([label_size, label_size, 1])
      
      single_input_sequence.append(sub_input)
      single_label_sequence.append(sub_label)

  return [np.array(single_input_sequence), np.array(single_label_sequence)]


manager = multiprocessing.Manager()

def initializer():
  pass
  
  


def print_preprocess_results(shared_dict,np_input):
  print("Finished preprocessing.")
  print(f'Total surface: {shared_dict["surf_photo"]+shared_dict["surf_render"]} pixels.')
  print(f'Photo surface: {shared_dict["surf_photo"]} pixels.')
  print(f'Rendered surface: {shared_dict["surf_render"]} pixels.')
  print(f'{shared_dict["surf_render"] / (shared_dict["surf_photo"]+shared_dict["surf_render"])*100}% of surface is rendered images.')
  print(f'Done, dataset len: {np_input.shape[0]}')


def multiprocess_train_setup(config):
  """
  Spawns several processes to pre-process the data
  """
  try:
    if not config.rebuild_dataset:
      input_cache = Path(config.checkpoint_dir) / Path("np_input.dat.npy")
      label_cache = Path(config.checkpoint_dir) / Path("np_label.dat.npy")
      print(f"Searching for cached dataset at {input_cache} and {label_cache}.")
      np_input = np.load(input_cache)
      np_label = np.load(label_cache)
      print("Loaded from cache. Dataset len:",np_input.shape[0])
      return (np_input,np_label)
  except Exception as e:
    pass
  
  print("Generating dataset...")

  data = prepare_data(dataset=config.train_dir)
  shared_dict = manager.dict()
  shared_dict_lock = manager.Lock()
  shared_dict["preprocessed"] = 0
  shared_dict["surf_photo"] = 0
  shared_dict["surf_render"] = 0

  print(f'{len(data)} images to preprocess.')

  with multiprocessing.Pool(multiprocessing.cpu_count() - 1,initializer,()) as pool:
    config_values = [config.image_size, config.label_size, config.stride, config.scale, config.padding // 2, config.distort]
    results = pool.map(train_input_worker, [(shared_dict,shared_dict_lock,data[i], config_values) for i in range(len(data))] )
  print(f'Done preprocessing, starting consolidation process.')
  
  print("Writing individual entries to disk...")
  result_c = len(results)
  entries_c = 0
  for num,result in enumerate(results):
    input_r,  label_r = result
    input_shape = input_r.shape
    label_shape = label_r.shape
    np.save(f"TEMP_{num}_i",input_r)
    np.save(f"TEMP_{num}_l",label_r)
    entries_c += input_r.shape[0]
  print("Done.")
  
  #free memory for large datasets  
  import gc
  del results
  gc.collect()
  
  #loading into a single array  
  print("Allocating array...")
  np_input = np.empty((entries_c,input_shape[1],input_shape[2],input_shape[3]))
  np_label = np.empty((entries_c,label_shape[1],label_shape[2],label_shape[3]))
  print("Loading array and removing temp files...")
  pos = 0
  for num in range(result_c):
     input_r = np.load(f"TEMP_{num}_i.npy")
     os.remove(f"TEMP_{num}_i.npy")
     label_r = np.load(f"TEMP_{num}_l.npy")
     os.remove(f"TEMP_{num}_l.npy")
     np_input[pos:pos+input_r.shape[0],:,:,:] = input_r
     np_label[pos:pos+input_r.shape[0],:,:,:] = label_r
     pos += input_r.shape[0]

  assert(pos == entries_c) #check we have all the files

  print_preprocess_results(shared_dict,np_input)

  print(f'Saving dataset files in "{Path(config.checkpoint_dir)}".')
  np.save(Path(config.checkpoint_dir) / Path("np_input.dat"),np_input)
  np.save(Path(config.checkpoint_dir) / Path("np_label.dat"),np_label)
  return (np_input,np_label)

def test_input_setup(config):
  # Load data path
  data_list = prepare_data(dataset=config.test_dir)

  shared_dict = manager.dict()
  shared_dict_lock = manager.Lock()
  shared_dict["preprocessed"] = 0
  shared_dict["surf_photo"] = 0
  shared_dict["surf_render"] = 0

  input_list = []
  label_list = []

  for data in data_list:
    input_, label_ = preprocess(shared_dict,shared_dict_lock,data, config.scale, config.distort)

    config_values = [config.image_size, config.label_size, config.stride, config.scale, config.padding // 2, config.distort]
    arrdata, arrlabel = preprocess_and_cut(shared_dict,shared_dict_lock,data, config_values)

    input_list.append(arrdata)
    label_list.append(arrlabel)

  input_list = np.concatenate(input_list)
  label_list = np.concatenate(label_list)
  
  print_preprocess_results(shared_dict,input_list)

  return (input_list, label_list)


def merge(config, og_image, upscaled_y_array):
  """
  Merges super-resolved image with original chroma components
  """
  (og_width, og_height) = og_image.size
  upscaled_og = og_image.convert('YCbCr').resize((og_width * config.scale, og_height * config.scale), Image.BICUBIC)
  (upscaled_width, upscaled_height) = upscaled_og.size
  crop_width = upscaled_width-upscaled_y_array.shape[1]
  crop_height = upscaled_height-upscaled_y_array.shape[0]
  upscaled_og = upscaled_og.crop((crop_width/2,crop_height/2,upscaled_width-crop_width/2,upscaled_height-crop_height/2))
  (width, height) = upscaled_og.size
  CbCr = np.frombuffer(upscaled_og.tobytes(), dtype=np.uint8).reshape(height, width, 3)[:,:,1:]
  upscaled_y_array = upscaled_y_array.round().astype(np.uint8)
  img = np.concatenate((upscaled_y_array, CbCr), axis=-1)
  return Image.fromarray(img.astype('uint8'), 'YCbCr').convert('RGB')



def fix_names(name):
  assert(len(name.split('/')) == 2)
  name, extension = name.split('/')
  if not 'wb' in name:
    return name
  if extension == "bias":
    return name.replace("wb","b")
  elif extension == "kernel":
    return name.replace("wb","w")
  else:
    assert(1)


def flatten(xss):
  return [x for xs in xss for x in xs]

def save_params(model, params, name):
  param_dir = "params/"

  if not os.path.exists(param_dir):
    os.makedirs(param_dir)

  filename = param_dir + f"weights_{'_'.join(str(i) for i in params)}.{name}.txt"
  h = open(filename, 'w')


  variables = {weight.path: weight.value.numpy() for weight in model.weights}

  for name, weights in variables.items():
    print(f' --- name={name} weights={weights} --- \n')
    h.write("{} =\n".format(fix_names(name)))

    if len(weights.shape) < 4:
        h.write("{}\n\n".format(weights.flatten().tolist()))
    else:
        h.write("[")
        sep = False
        for filter_x in range(len(weights)):
          for filter_y in range(len(weights[filter_x])):
            filter_weights = weights[filter_x][filter_y]
            for input_channel in range(len(filter_weights)):
              for output_channel in range(len(filter_weights[input_channel])):
                val = filter_weights[input_channel][output_channel]
                if sep:
                    h.write(', ')
                h.write("{}".format(val))
                sep = True
              h.write("\n  ")
        h.write("]\n\n")

  h.close()
  print(f'Saved weights to "{filename}".')
