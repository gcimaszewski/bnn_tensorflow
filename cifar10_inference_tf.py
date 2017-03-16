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

import hardware_net
import FixedPoint
import Printer

from pylearn2.datasets.cifar10 import CIFAR10

if __name__ == "__main__":
#def conv_check():

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
    activation = hardware_net.SignTheano
    print("activation = sign(x)")

    no_bias = True
    print("no_bias = " + str(no_bias))

    # BinaryConnect
    H = 1.
    print('Loading CIFAR-10 dataset...')

    test_set = CIFAR10(which_set="test")
    print("Test set size = "+str(len(test_set.X)))
    test_instances = 1000
    print("Using instances 0 .. "+str(test_instances))

    # bc01 format
    # Inputs in the range [-1,+1]
    test_set.X = np.reshape(np.subtract(np.multiply(2./255.,test_set.X),1.),(-1,3,32,32))
  #  test_set.X = test_set.X.transpose([0, 2, 3, 1])
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
    # X = FixedPoint.FixedPoint(32,31).convert(X)
    y = test_set.y[0:test_instances]


    print('Building the CNN...')

    #the input layer
    #tensorflow doesn't have an OO format for layers - layers are just functions
    inputs = tf.placeholder(tf.float32, [1, 32, 32, 3])
    inputs_tr = tf.placeholder(tf.float32, [None, 32, 32, 3])
    #target = T.matrix(X, 'targets')

    #final output from the nonlinearity layer
 #   target = tf.placeholder(tf.float32, [None, 10])
    conv_result = tf.placeholder(tf.float32, [None, 32, 32, 128])
    weight1 = tf.Variable(tf.zeros([128, 3, 3, 3]))
    weight2 = tf.Variable(tf.zeros([128, 128, 3, 3]))
    weight3 = tf.Variable(tf.zeros([256, 128, 3, 3]))
    weight4 = tf.Variable(tf.zeros([256, 256, 3, 3]))
    weight5 = tf.Variable(tf.zeros([512, 256, 3, 3]))
    weight6 = tf.Variable(tf.zeros([512, 512, 3, 3]))
    weight7 = tf.Variable(tf.zeros([8192, 1024]))
    weight8 = tf.Variable(tf.zeros([1024, 1024]))
    weight9 = tf.Variable(tf.zeros([1024, 10]))

    #format of weight in tf: [inputdim, inputdim, numchannelin, numchannelout]


    k1 = tf.Variable(tf.zeros([128]))
    h1 = tf.Variable(tf.zeros([128]))
    k2 = tf.Variable(tf.zeros([128]))
    h2 = tf.Variable(tf.zeros([128]))
    k3 = tf.Variable(tf.zeros([256]))
    h3 = tf.Variable(tf.zeros([256]))
    k4 = tf.Variable(tf.zeros([256]))
    h4 = tf.Variable(tf.zeros([256]))
    k5 = tf.Variable(tf.zeros([512]))
    h5 = tf.Variable(tf.zeros([512]))
    k6 = tf.Variable(tf.zeros([512]))
    h6 = tf.Variable(tf.zeros([512]))
    k7 = tf.Variable(tf.zeros([1024]))
    h7 = tf.Variable(tf.zeros([1024]))
    k8 = tf.Variable(tf.zeros([1024]))
    h8 = tf.Variable(tf.zeros([1024]))
    k9 = tf.Variable(tf.zeros([10]))
    h9 = tf.Variable(tf.zeros([10]))

    #try transposing the weight feature maps

    def cnn(x, weights, h, k):

        #filter is only weight
        weight1 = tf.transpose(weights[0], [2, 3,1,0])
        x = tf.nn.conv2d(x, weight1, [1,1,1,1], "SAME")
        x = hardware_net_tf.batch_norm(x, h[0], k[0])
        x = hardware_net_tf.SignTheano(x)

        weight2 = tf.transpose(weights[1], [2, 3, 1,0])
        x = tf.nn.conv2d(x, weight2, [1, 1, 1,1], "SAME")
        x = tf.contrib.layers.max_pool2d(x, kernel_size = [2,2])
        x = hardware_net_tf.batch_norm(x, h[1], k[1])
        x = hardware_net_tf.SignTheano(x)


        #third convolution
        weight3 = tf.transpose(weights[2], [2, 3, 1,0])
        x = tf.nn.conv2d(x, weight3, [1,1,1,1], "SAME")
        x = hardware_net_tf.batch_norm(x, h[2], k[2])
        x = hardware_net_tf.SignTheano(x)

        #fourth convolution
        weight4 = tf.transpose(weights[3], [2, 3, 1,0])
        x = tf.nn.conv2d(x, weight4, [1,1,1,1], "SAME")
        x = tf.contrib.layers.max_pool2d(x, kernel_size = [2,2])
        x = hardware_net_tf.batch_norm(x, h[3], k[3])
        x = hardware_net_tf.SignTheano(x)

        #fifth convolution
        weight5 = tf.transpose(weights[4], [2,3,1,0])
        x = tf.nn.conv2d(x, weight5, [1,1,1,1], "SAME")
        x = hardware_net_tf.batch_norm(x, h[4], k[4])
        x = hardware_net_tf.SignTheano(x)

        #sixth and final convolution
        weight6 = tf.transpose(weights[5], [2,3,1,0])
        x = tf.nn.conv2d(x, weight6, [1,1,1,1], "SAME")
        x = tf.contrib.layers.max_pool2d(x, kernel_size = [2,2])
        x = hardware_net_tf.batch_norm(x, h[5], k[5])
        x = hardware_net_tf.SignTheano(x)

        x = tf.transpose(x, [0, 3, 1, 2])

        rand_wts_1 = tf.random_uniform([4*4*512, 1024], minval=-1.0, maxval=1.0)
        rand_wts_2 = tf.random_uniform([1024, 1024], minval=-1.0, maxval=1.0)
        rand_wts_3 = tf.random_uniform([1024, 10], minval=-1.0, maxval=1.0)



        x = tf.reshape(x, [1, 4*4*512])
      #  rand_wts_1 = tf.random_uniform([4*4*512, 1024], minval=-1.0, maxval=1.0)
        x = tf.matmul(x, weights[6])
        #x = tf.layers.dense(x, units=1024, kernel_initializer = rand_wts_1, bias_regularizer = None)
        x = hardware_net_tf.batch_norm(x, h[6], k[6])
        x = hardware_net_tf.SignTheano(x)

        # #x = tf.layers.dense(x, units=1024)
        x = tf.matmul(x, weights[7])
        x = hardware_net_tf.batch_norm(x, h[7], k[7])
        x = hardware_net_tf.SignTheano(x)

        # #final dense layer
        # #x = tf.layers.dense(x, units = 10)
        x = tf.matmul(x, weights[8])
        x = hardware_net_tf.batch_norm(x, h[8], k[8])
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
          #  param.set_value(hardware_net.SignNumpy(param_values[param])
          #  param_values_mod[param] = hardware_net_tf.SignNumpy(param_values[param])
            weights_arr[param/3] = hardware_net_tf.SignNumpy(param_values[param])
            #if l == lout:
           #     print "kernel 0 for layer", l
            #    Printer.print_3d(param_values_mod[param][0,:,:,:], 3, 3, 3, 'b')
        #k
        elif (param%3) == 1:
          #  param.set_value(k_fix.convert(param_values[param])
           # param_values_mod[param] = k_fix.convert(param_values[param])


            #k_arr[param/3] = hardware_net_tf.SignNumpy(param_values[param])
            k_arr[param/3] = k_fix.convert((param_values[param]))
          #  if l == lout:
      #          print "k =", param_values_mod[param][0:4]
        #h
        elif (param%3) == 2:
      #      param_values_mod[param] = h_fix.convert(param_values[param])


            #h_arr[param/3] = hardware_net_tf.SignNumpy(param_values[param])
            h_arr[param/3] = h_fix.convert(param_values[param])

        #    if l == lout:
        #        print "h =", param_values_mod[param][0:4]
        #    l = l + 1
        else:
            print "Incorrect param name", param.name
            exit(-1)



    #try reformatting the weights to see if conv works
    for conv in range(6):
        for num_input_maps in range(len(weights_arr[conv])):
            for num_output_maps in range(len(weights_arr[conv][num_input_maps])):
                 #   print  weights_arr[conv][num_input_maps][num_output_maps]
                    weights_arr[conv][num_input_maps][num_output_maps] = np.flip(weights_arr[conv][num_input_maps][num_output_maps], 1)
                    weights_arr[conv][num_input_maps][num_output_maps] = np.flip(weights_arr[conv][num_input_maps][num_output_maps], 0)


    def calculate_error(X, target, output):
      #  test_loss = np.mean(np.sqrt(np.max(0., (1. - target*output))))
        test_err = np.mean(np.not_equal(np.argmax(output, axis=1), np.argmax(target, axis=1)))
        return test_err



    print('Running...')

    start_time = time.time()
    h_total = [h1, h2, h3, h4, h5, h6, h7, h8, h9]
    k_total = [k1, k2, k3, k4, k5, k6, k7, k8, k9]
    w_total = [weight1, weight2, weight3, weight4, weight5, weight6, weight7, weight8, weight9]
    output = []

    conv_output = cnn(inputs, w_total, h_total, k_total)

    init = tf.global_variables_initializer()

    start_time = time.time()


    with tf.Session() as sess:  ##np.transpose(weights_arr[0], (1, 2, 3, 0))
        sess.run(init)
        steps = 0
        while steps < test_instances:
            result = sess.run(conv_output, feed_dict={weight1: weights_arr[0], weight2: weights_arr[1], weight3:weights_arr[2], weight4:weights_arr[3],\
                weight5: weights_arr[4], weight6:weights_arr[5], weight7:weights_arr[6], weight8: weights_arr[7], weight9:weights_arr[8], k1:k_arr[0], h1: h_arr[0], h2: h_arr[1], k2: k_arr[1], k3: k_arr[2], h3: h_arr[2], \
                k4:k_arr[3], h4:h_arr[3], k5:k_arr[4], h5:h_arr[4], k6:k_arr[5], h6:h_arr[5], \
                k7:k_arr[6], h7:h_arr[6], k8:k_arr[7], h8:h_arr[7], k9:k_arr[8], h9:h_arr[8], inputs: [X[steps]]})
            steps += 1
            output.append(result[0])
            #output.append(result[0].flatten())


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


