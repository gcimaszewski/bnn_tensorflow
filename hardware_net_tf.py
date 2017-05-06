import time
import numpy as np

import tensorflow as tf
#import tf.layers
from tensorflow.python.framework import ops



# We will not use any of these layers in training, so we can cut a lot
# of corners

# This sends a distribution between -1 and 1 to 0 and 1, scaling by 0.5 and
# adding +0.5 y-offset
def hard_sigmoid(x):
  #  return T.clip((x+1.)/2., 0,1)
  return tf.clip_by_value((x+1.)/2, 0, 1)

# Weight binarization function
def SignNumpy(x):
    return np.float32(2.*np.greater_equal(x,0)-1.)

# Activation binarization function
def SignTheano(x):
 
    return tf.subtract(tf.multiply(tf.cast(tf.greater_equal(x, tf.zeros(tf.shape(x))), tf.float32), 2.0), 1.0)



# The weights' binarization function, 
# taken directly from the BinaryConnect github repository and simplified
# (which was made available by his authors)
def binarization(W, H, binary=True):

    if not binary:
        Wb = W
    else:
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        Wb = T.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    return Wb

def batch_norm(x, h, k):

    # axes = (0,) + tuple(range(2, len(x.get_shape())))
    # param_axes = iter(range(x.get_shape().ndims - len(axes)))
    # pattern = ['x' if input_axis in axes \
    #             else next(param_axes) \
    #             for input_axis in range(x.get_shape().ndims)]

    # broadcast=[]
    # inds = []
    # for index in range(len(pattern)):
    #     if pattern[index]=='x':
    #         broadcast.append(index)
    #     else:
    #         inds.append(index)

    # broadcast = [0,1,2]

    # #different order of axes
    # # should give['x', 'x', 'x', 0]
    # #x=true
    # if True:
    #     for ind in broadcast:
    #         h = tf.expand_dims(h, ind)
    #         k = tf.expand_dims(k, ind)

    # else:
    #     h = tf.expand_dims(tf.transpose(h, inds), broadcast)
    #     k = tf.expand_dims(tf.transpose(k, inds), broadcast)

    #    normalized = input*k + h
    normalized = tf.multiply(x, k)
    normalized = tf.add(normalized, h)
    return normalized


# class conv2d_layer(tf.layers.conv2d):
#     def __init__(self):
#         tf.layers.conv2d.__init__(self)






