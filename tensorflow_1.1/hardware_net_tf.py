import time
import numpy as np

import tensorflow as tf
from tensorflow.python.framework import ops



# We will not use any of these layers in training, so we can cut a lot
# of corners

# This sends a distribution between -1 and 1 to 0 and 1, scaling by 0.5 and
# adding +0.5 y-offset
def hard_sigmoid(x):
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
        Wb = tf.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = tf.cast(tf.where(Wb,H,-H), tf.float)
    
    return Wb

def batch_norm(x, h, k):

    #modified batch norm operation as described in paper
    normalized = tf.multiply(x, k)
    normalized = tf.add(normalized, h)
    return normalized







