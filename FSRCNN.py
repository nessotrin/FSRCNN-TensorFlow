from tensorflow.keras.layers import Conv2D, Input, PReLU, Conv2DTranspose
from tensorflow.keras.ops import clip
from tensorflow.keras.initializers import RandomNormal, HeNormal
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.layers import Lambda


class FSRCNNModel():

  def __init__(self,scale):
    self.scale = scale
    self.d = 12
    self.s = 0
    self.m = 4
    self.r = 1

    self.radius = 1
    self.name = "FSRCNNX"

  def getParams(self):
    return [self.scale, self.d, self.s, self.m, self.r]

  def getModel(self):
    d = self.d 
    s = self.s #0 => s=d
    m = self.m

    #feature layer
    X_in = Input(shape=(None, None, 1))
    X = Conv2D(filters=d, kernel_size=5, padding='valid',
               kernel_initializer=HeNormal(), name = 'wb1')(X_in)


    #shrinking layer, disabled with s = 0
    if s > 0:
      X = PReLU(shared_axes=[1, 2], name = 'alpha1')(X)
      X = Conv2D(filters=s, kernel_size=1, padding='same',
                kernel_initializer=HeNormal(), name = 'wb2')(X)

    features = X
    s_or_d = s if s != 0 else d

    #mapping layer    
    mapping_count = 3
    for n in range(0, m):
      if n > 0:
        X = PReLU(shared_axes=[1, 2], name = f'alpha{mapping_count}')(X)
      X = Conv2D(filters=s_or_d, kernel_size=3, padding='same',
                   kernel_initializer=HeNormal(), name = f'wb{mapping_count}')(X)
      mapping_count +=1
      #sub-band residuals
      if n == m-1:
        X = PReLU(shared_axes=[1, 2], name = f'alpha{mapping_count}')(X)
        X = Conv2D(filters=s_or_d, kernel_size=1, padding='same',
                    kernel_initializer=HeNormal(), name = f'wb{mapping_count}')(X)
        X = X+features
        mapping_count +=1

    X = PReLU(shared_axes=[1, 2], name = 'alpha2')(X)

    # Expanding
    if s > 0:
      X = Conv2D(filters=d, kernel_size=1, padding='same',
                kernel_initializer=HeNormal(), name = f'wb{mapping_count}')(X)
      X = PReLU(shared_axes=[1, 2], name = f'alpha{mapping_count}')(X)


    #Sub-pixel convolution
    new_size = self.radius * 2 + 1
    X = Conv2D(filters=self.scale**2, kernel_size=new_size, padding='same',
              kernel_initializer=RandomNormal(), name = f'deconv_wb')(X)
    if self.scale > 1:
      Subpixel_layer = Lambda(lambda x:tf.nn.depth_to_space(x,self.scale))
      X = Subpixel_layer(inputs=X)

    X_out = clip(X, 0.0, 1.0)


    return Model(X_in, X_out)

  def loss(self, Y, X):
    X = tf.cast(X, tf.float64)
    Y = tf.cast(Y, tf.float64)
    dY = tf.image.sobel_edges(Y)
    dX = tf.image.sobel_edges(X)
    M = tf.sqrt(tf.square(dY[:,:,:,:,0]) + tf.square(dY[:,:,:,:,1]))
    return tf.losses.MAE(dY, dX) \
         + tf.expand_dims(tf.losses.MAE((1.0 - M) * Y, (1.0 - M) * X) * 2.0,axis=-1)


