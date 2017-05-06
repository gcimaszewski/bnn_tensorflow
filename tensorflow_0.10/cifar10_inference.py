#rewrite using 1.0
#export to github
#rewrite using new interface- accuracy

import sys
import os
import time

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1')

import tensorflow as tf
from tensorflow.python.framework import ops
import hardware_net_tf

import cPickle as pickle
import gzip

import FixedPoint
import Printer

from pylearn2.datasets.cifar10 import CIFAR10
from collections import OrderedDict

if __name__ == "__main__":

    # BN parameters
    # alpha is the exponential moving average factor
    alpha = .1
    epsilon = 1e-4

    # Parameters directory
    if not os.environ.has_key('CRAFT_BNN_ROOT'):
        print "CRAFT_BNN_ROOT not set!"
        exit(-1)
    top_dir = os.environ['CRAFT_BNN_ROOT']
    params_dir = top_dir + '/params'

    # BinaryOut
    print("activation = sign(x)")

    no_bias = True
    print("no_bias = " + str(no_bias))

    # BinaryConnect
    H = 1.
    print('Loading CIFAR-10 dataset...')

    test_set = CIFAR10(which_set="test")
    print("Test set size = "+str(len(test_set.X)))
    test_instances = 10000
    print("Using instances 0 .. "+str(test_instances))

    # bc01 format
    # Inputs in the range [-1,+1]
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
    # flatten targets
    test_set.y = np.hstack(test_set.y)
    # Onehot the targets
    test_set.y = np.float32(np.eye(10)[test_set.y])
    # for hinge loss
    test_set.y = 2* test_set.y - 1.

    # print('Quantizing the input...')
    X = test_set.X[0:test_instances]
    #account for the data dimensions in tensorflow: ()
    X = np.transpose(X, [0, 2, 3, 1])
    y = test_set.y[0:test_instances]


    print('Building the CNN...')

    #the input layer
    #tensorflow doesn't have an OO format for layers - layers are just functions
    inputs = tf.placeholder(tf.float32, [1, 32, 32, 3])
    inputs_tr = tf.placeholder(tf.float32, [None, 32, 32, 3])

    conv_result = tf.placeholder(tf.float32, [None, 32, 32, 128])

    weights = {
        'wt1' : tf.Variable(tf.zeros([128, 3, 3, 3])),

        'wt2' : tf.Variable(tf.zeros([128, 128, 3, 3])), 

        'wt3' : tf.Variable(tf.zeros([256, 128, 3, 3])),

        'wt4' : tf.Variable(tf.zeros([256, 256, 3, 3])),

        'wt5' : tf.Variable(tf.zeros([512, 256, 3, 3])),

        'wt6' : tf.Variable(tf.zeros([512, 512, 3, 3])),

        'wt7' : tf.Variable(tf.zeros([8192, 1024])),

        'wt8' : tf.Variable(tf.zeros([1024, 1024])),

        'wt9' : tf.Variable(tf.zeros([1024, 10]))

    }

    #format of weight in tf: [inputdim, inputdim, numchannelin, numchannelout]


    scale = {

        'k1' : tf.Variable(tf.zeros([128])), 

        'k2' : tf.Variable(tf.zeros([128])),

        'k3' : tf.Variable(tf.zeros([256])),

        'k4' : tf.Variable(tf.zeros([256])),

        'k5' : tf.Variable(tf.zeros([512])),

        'k6' : tf.Variable(tf.zeros([512])),

        'k7' : tf.Variable(tf.zeros([1024])),

        'k8' : tf.Variable(tf.zeros([1024])),

        'k9' : tf.Variable(tf.zeros([10]))
    }

    offset = {

        'h1' : tf.Variable(tf.zeros([128])),

        'h2' : tf.Variable(tf.zeros([128])),

        'h3' : tf.Variable(tf.zeros([256])),

        'h4' : tf.Variable(tf.zeros([256])),

        'h5' : tf.Variable(tf.zeros([512])),

        'h6' : tf.Variable(tf.zeros([512])),

        'h7' : tf.Variable(tf.zeros([1024])),

        'h8' : tf.Variable(tf.zeros([1024])),

        'h9' : tf.Variable(tf.zeros([10]))

    }


    def cnn(x, weights, scale, offset):

        weight1 = tf.transpose(weights['wt1'], [2, 3,1,0])
        x = tf.nn.conv2d(x, weight1, [1,1,1,1], "SAME")
        x = hardware_net_tf.batch_norm(x, offset['h1'], scale['k1'])
        x = hardware_net_tf.SignTheano(x)

        weight2 = tf.transpose(weights['wt2'], [2, 3, 1,0])
        x = tf.nn.conv2d(x, weight2, [1, 1, 1,1], "SAME")
        x = tf.contrib.layers.max_pool2d(x, kernel_size = [2,2])
        x = hardware_net_tf.batch_norm(x, offset['h2'], scale['k2'])
        x = hardware_net_tf.SignTheano(x)

        weight3 = tf.transpose(weights['wt3'], [2, 3, 1,0])
        x = tf.nn.conv2d(x, weight3, [1,1,1,1], "SAME")
        x = hardware_net_tf.batch_norm(x, offset['h3'], scale['k3'])
        x = hardware_net_tf.SignTheano(x)

        weight4 = tf.transpose(weights['wt4'], [2, 3, 1,0])
        x = tf.nn.conv2d(x, weight4, [1,1,1,1], "SAME")
        x = tf.contrib.layers.max_pool2d(x, kernel_size = [2,2])
        x = hardware_net_tf.batch_norm(x, offset['h4'], scale['k4'])
        x = hardware_net_tf.SignTheano(x)

        weight5 = tf.transpose(weights['wt5'], [2,3,1,0])
        x = tf.nn.conv2d(x, weight5, [1,1,1,1], "SAME")
        x = hardware_net_tf.batch_norm(x, offset['h5'], scale['k5'])
        x = hardware_net_tf.SignTheano(x)

        weight6 = tf.transpose(weights['wt6'], [2,3,1,0])
        x = tf.nn.conv2d(x, weight6, [1,1,1,1], "SAME")
        x = tf.contrib.layers.max_pool2d(x, kernel_size = [2,2])
        x = hardware_net_tf.batch_norm(x, offset['h6'], scale['k6'])
        x = hardware_net_tf.SignTheano(x)

        x = tf.transpose(x, [0, 3, 1, 2])

        x = tf.reshape(x, [1, 4*4*512])
        x = tf.matmul(x, weights['wt7'])
        x = hardware_net_tf.batch_norm(x, offset['h7'], scale['k7'])
        x = hardware_net_tf.SignTheano(x)

        x = tf.matmul(x, weights['wt8'])
        x = hardware_net_tf.batch_norm(x, offset['h8'], scale['k8'])
        x = hardware_net_tf.SignTheano(x)

        x = tf.matmul(x, weights['wt9'])
        x = hardware_net_tf.batch_norm(x, offset['h9'], scale['k9'])
        x = hardware_net_tf.SignTheano(x)
        return x




    print("Loading the trained parameters and binarizing the weights...")
    num_layers = 9

    # Load parameters
    with np.load(params_dir + '/cifar10_parameters_nb.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(num_layers * 3)]
 
    k_fix = FixedPoint.FixedPoint(16,15)
    h_fix = FixedPoint.FixedPoint(16,12)

    # Binarize the weights
    l = 1
    lout = 1
    weights_arr = [None for i in range(num_layers)]
    k_arr = [None for i in range(num_layers)]
    h_arr = [None for i in range(num_layers)]

    for param in range(num_layers*3):
        if (param%3) == 0:

            weights_arr[param/3] = hardware_net_tf.SignNumpy(param_values[param])

        elif (param%3) == 1:

            k_arr[param/3] = k_fix.convert((param_values[param]))

        elif (param%3) == 2:

            h_arr[param/3] = h_fix.convert(param_values[param])

        else:
            print "Incorrect param name", param.name
            exit(-1)



    #try reformatting the weights to see if conv works
    for conv in range(6):
        for num_input_maps in range(len(weights_arr[conv])):
            for num_output_maps in range(len(weights_arr[conv][num_input_maps])):
                    weights_arr[conv][num_input_maps][num_output_maps] = np.flip(weights_arr[conv][num_input_maps][num_output_maps], 1)
                    weights_arr[conv][num_input_maps][num_output_maps] = np.flip(weights_arr[conv][num_input_maps][num_output_maps], 0)


    def calculate_error(X, target, output):
      #  test_loss = np.mean(np.sqrt(np.max(0., (1. - target*output))))
        test_err = np.mean(np.not_equal(np.argmax(output, axis=1), np.argmax(target, axis=1)))
        return test_err



    print('Running...')

    output = []

    values_dict={}
    for i in range(num_layers):
        values_dict[weights['wt'+str(i+1)], scale['k'+str(i+1)], offset['h'+str(i+1)]] =(weights_arr[i],k_arr[i],h_arr[i])

    conv_output = cnn(inputs, weights, scale, offset)

    init = tf.initialize_all_variables()

    start_time = time.time()

    with tf.Session() as sess:  
        sess.run(init)
        steps = 0
        while steps < test_instances:
            values_dict[inputs]= X[[steps]]
            result = sess.run(conv_output, feed_dict=values_dict)
            steps += 1
            output.append(result[0])


    # input_ = input_fn(X)
    # output = cnn_fn(X)
    # conv_out = conv_fn(X)
    # print "input shape=", input_.shape
    # Printer.print_2d(input_[0,0,:,:], 8, 8, 'b')
    # print " ##"
    # Printer.print_2d(input_[0,1,:,:], 8, 8, 'b')

    # print "\nconv shape=", conv_out.shape
    # Printer.print_2d(conv_out[0,0,:,:], 8, 8, 'i')

    # print "\noutput shape=", output.shape
    # Printer.print_2d(output[0,0,:,:], 8, 8, 'b')
    # print " ##"
    # Printer.print_2d(output[0,1,:,:], 8, 8, 'b')
    # print " ##"
    # Printer.print_2d(output[0,2,:,:], 8, 8, 'b')

    #np.savez("py_conv2_maps.npz", output);


    print "error = " +str(calculate_error(X, y, output))
    run_time = time.time() - start_time
    print("run_time = "+str(run_time)+"s")


