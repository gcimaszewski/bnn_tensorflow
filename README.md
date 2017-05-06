# bnn_tensorflow
## Description
Included is the tensorflow translation of the original Theano/lasagne BNN inference code, based on the BNN paper by Courbariaux et al.  
In my test runs, it produces an error of 12.08%.  
Versions for both the older 0.10 and current 1.1 CPU-only versions of tensorflow are included. 
Both versions are run as scripts, i.e. by calling "python cifar10_inference.py". 


### Requirements 

1. numpy
2. pylearn2

### Troubleshooting
Note: I had problems importing the CIFAR10 dataset from pylearn2.  I don't know if this was merely due to my directory setup, but I was able to fix it with: 
```export PYLEARN2_DATA_PATH=$HOME/pylearn2/pylearn2/scripts/datasets/```
