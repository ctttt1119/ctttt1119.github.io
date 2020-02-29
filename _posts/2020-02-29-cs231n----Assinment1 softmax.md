---
layout:     post
title:      cs231n
subtitle:   Softmax
date:       2020-02-29
author:     Shawn
header-img: img/Stanford.jpg
catalog: true
tags:
    - cs231n













---

# Softmax exercise

*Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*

This exercise is analogous to the SVM exercise. You will:

- implement a fully-vectorized **loss function** for the Softmax classifier
- implement the fully-vectorized expression for its **analytic gradient**
- **check your implementation** with numerical gradient
- use a validation set to **tune the learning rate and regularization** strength
- **optimize** the loss function with **SGD**
- **visualize** the final learned weights



```python
import random
import numpy as np
from cs231n.data_utils import load_CIFAR10
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading extenrnal modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2
```


```python
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    
    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
       del X_train, y_train
       del X_test, y_test
       print('Clear previously loaded data.')
    except:
       pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]
    
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image
    
    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('dev data shape: ', X_dev.shape)
print('dev labels shape: ', y_dev.shape)
```

    Train data shape:  (49000, 3073)
    Train labels shape:  (49000,)
    Validation data shape:  (1000, 3073)
    Validation labels shape:  (1000,)
    Test data shape:  (1000, 3073)
    Test labels shape:  (1000,)
    dev data shape:  (500, 3073)
    dev labels shape:  (500,)


## Softmax Classifier

Your code for this section will all be written inside **cs231n/classifiers/softmax.py**. 



```python
# First implement the naive softmax loss function with nested loops.
# Open the file cs231n/classifiers/softmax.py and implement the
# softmax_loss_naive function.

from cs231n.classifiers.softmax import softmax_loss_naive
import time

# Generate a random softmax weight matrix and use it to compute the loss.
W = np.random.randn(3073, 10) * 0.0001
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# As a rough sanity check, our loss should be something close to -log(0.1).
print('loss: %f' % loss)
print('sanity check: %f' % (-np.log(0.1)))
```

    loss: 2.399115
    sanity check: 2.302585


**Inline Question 1**

Why do we expect our loss to be close to -log(0.1)? Explain briefly.**

$\color{blue}{\textit Your Answer:}$ *Fill this in* 




```python
# Complete the implementation of softmax_loss_naive and implement a (naive)
# version of the gradient that uses nested loops.
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 0.0)

# As we did for the SVM, use numeric gradient checking as a debugging tool.
# The numeric gradient should be close to the analytic gradient.
from cs231n.gradient_check import grad_check_sparse
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.0)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)

# similar to SVM case, do another gradient check with regularization
loss, grad = softmax_loss_naive(W, X_dev, y_dev, 5e1)
f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 5e1)[0]
grad_numerical = grad_check_sparse(f, W, grad, 10)
```

    numerical: 0.807611 analytic: 0.807611, relative error: 5.313372e-08
    numerical: -2.705861 analytic: -2.705861, relative error: 1.295587e-09
    numerical: 0.408547 analytic: 0.408547, relative error: 2.210474e-07
    numerical: -1.900769 analytic: -1.900768, relative error: 1.883594e-08
    numerical: 1.229252 analytic: 1.229252, relative error: 3.474095e-08
    numerical: 2.742905 analytic: 2.742905, relative error: 1.580066e-08
    numerical: -5.390659 analytic: -5.390659, relative error: 6.167341e-09
    numerical: 2.891738 analytic: 2.891738, relative error: 2.651216e-09
    numerical: -0.649408 analytic: -0.649408, relative error: 1.483085e-08
    numerical: 2.193099 analytic: 2.193098, relative error: 4.603736e-08
    numerical: -1.531906 analytic: -1.531906, relative error: 3.621779e-08
    numerical: 2.647067 analytic: 2.647067, relative error: 3.911318e-08
    numerical: -1.902443 analytic: -1.902443, relative error: 4.499142e-08
    numerical: -0.756685 analytic: -0.756685, relative error: 2.580978e-08
    numerical: 1.728975 analytic: 1.728975, relative error: 1.333948e-08
    numerical: 0.964926 analytic: 0.964926, relative error: 3.315943e-08
    numerical: -0.379811 analytic: -0.379811, relative error: 4.988555e-08
    numerical: -2.780767 analytic: -2.780768, relative error: 2.467740e-08
    numerical: 0.488085 analytic: 0.488085, relative error: 1.303925e-07
    numerical: -1.561384 analytic: -1.561384, relative error: 4.714033e-08



```python
# Now that we have a naive implementation of the softmax loss function and its gradient,
# implement a vectorized version in softmax_loss_vectorized.
# The two versions should compute the same results, but the vectorized version should be
# much faster.
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

from cs231n.classifiers.softmax import softmax_loss_vectorized
tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

# As we did for the SVM, we use the Frobenius norm to compare the two versions
# of the gradient.
grad_difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
print('Loss difference: %f' % np.abs(loss_naive - loss_vectorized))
print('Gradient difference: %f' % grad_difference)
```

    naive loss: 2.399115e+00 computed in 0.115638s
    vectorized loss: 2.399115e+00 computed in 0.002995s
    Loss difference: 0.000000
    Gradient difference: 0.000000



```python
# Use the validation set to tune hyperparameters (regularization strength and
# learning rate). You should experiment with different ranges for the learning
# rates and regularization strengths; if you are careful you should be able to
# get a classification accuracy of over 0.35 on the validation set.
from cs231n.classifiers import Softmax
results = {}
best_val = -1
best_softmax = None
learning_rates = [1e-7, 5e-7]
regularization_strengths = [2.5e4, 5e4]

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained softmax classifer in best_softmax.                          #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

for learning_rate in learning_rates:
    for regularization_strength in regularization_strengths:
        softmax = Softmax()
        loss_hist = softmax.train(X_train, y_train, learning_rate=learning_rate, reg=regularization_strength,num_iters=1500, verbose=True)
        y_train_pred = softmax.predict(X_train)
        y_val_pred = softmax.predict(X_val)
        train_accuracy = np.mean(y_train == y_train_pred)
        val_accuracy = np.mean(y_val == y_val_pred)
        results[(learning_rate,regularization_strength)] = [train_accuracy, val_accuracy]
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_softmax = softmax

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)
```

    iteration 0 / 1500: loss 387.065947
    iteration 100 / 1500: loss 233.803751
    iteration 200 / 1500: loss 142.250290
    iteration 300 / 1500: loss 86.731538
    iteration 400 / 1500: loss 53.294812
    iteration 500 / 1500: loss 33.069771
    iteration 600 / 1500: loss 20.803764
    iteration 700 / 1500: loss 13.249054
    iteration 800 / 1500: loss 8.833243
    iteration 900 / 1500: loss 6.278142
    iteration 1000 / 1500: loss 4.556531
    iteration 1100 / 1500: loss 3.552768
    iteration 1200 / 1500: loss 2.929662
    iteration 1300 / 1500: loss 2.514156
    iteration 1400 / 1500: loss 2.378498
    iteration 0 / 1500: loss 783.887543
    iteration 100 / 1500: loss 287.964768
    iteration 200 / 1500: loss 106.704931
    iteration 300 / 1500: loss 40.429433
    iteration 400 / 1500: loss 15.987971
    iteration 500 / 1500: loss 7.196672
    iteration 600 / 1500: loss 3.974838
    iteration 700 / 1500: loss 2.755741
    iteration 800 / 1500: loss 2.288230
    iteration 900 / 1500: loss 2.187884
    iteration 1000 / 1500: loss 2.203521
    iteration 1100 / 1500: loss 2.107022
    iteration 1200 / 1500: loss 2.082134
    iteration 1300 / 1500: loss 2.016539
    iteration 1400 / 1500: loss 2.110078
    iteration 0 / 1500: loss 391.281708
    iteration 100 / 1500: loss 32.955716
    iteration 200 / 1500: loss 4.476195
    iteration 300 / 1500: loss 2.263634
    iteration 400 / 1500: loss 2.008487
    iteration 500 / 1500: loss 2.011468
    iteration 600 / 1500: loss 2.075513
    iteration 700 / 1500: loss 2.030146
    iteration 800 / 1500: loss 2.037694
    iteration 900 / 1500: loss 2.013936
    iteration 1000 / 1500: loss 2.051557
    iteration 1100 / 1500: loss 2.047541
    iteration 1200 / 1500: loss 2.084142
    iteration 1300 / 1500: loss 1.999689
    iteration 1400 / 1500: loss 2.021448
    iteration 0 / 1500: loss 772.791248
    iteration 100 / 1500: loss 6.926631
    iteration 200 / 1500: loss 2.124671
    iteration 300 / 1500: loss 2.081373
    iteration 400 / 1500: loss 2.061357
    iteration 500 / 1500: loss 2.040888
    iteration 600 / 1500: loss 2.028865
    iteration 700 / 1500: loss 2.134159
    iteration 800 / 1500: loss 2.023190
    iteration 900 / 1500: loss 2.088824
    iteration 1000 / 1500: loss 2.055338
    iteration 1100 / 1500: loss 2.093477
    iteration 1200 / 1500: loss 2.072089
    iteration 1300 / 1500: loss 2.060196
    iteration 1400 / 1500: loss 2.120550
    lr 1.000000e-07 reg 2.500000e+04 train accuracy: 0.350286 val accuracy: 0.372000
    lr 1.000000e-07 reg 5.000000e+04 train accuracy: 0.327776 val accuracy: 0.343000
    lr 5.000000e-07 reg 2.500000e+04 train accuracy: 0.345959 val accuracy: 0.354000
    lr 5.000000e-07 reg 5.000000e+04 train accuracy: 0.334531 val accuracy: 0.354000
    best validation accuracy achieved during cross-validation: 0.372000



```python
# evaluate on test set
# Evaluate the best softmax on test set
y_test_pred = best_softmax.predict(X_test)
test_accuracy = np.mean(y_test == y_test_pred)
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy, ))
```

    softmax on raw pixels final test set accuracy: 0.360000


**Inline Question 2** - *True or False*

Suppose the overall training loss is defined as the sum of the per-datapoint loss over all training examples. It is possible to add a new datapoint to a training set that would leave the SVM loss unchanged, but this is not the case with the Softmax classifier loss.

$\color{blue}{\textit Your Answer:}$


$\color{blue}{\textit Your Explanation:}$




```python
# Visualize the learned weights for each class
w = best_softmax.W[:-1,:] # strip out the bias
w = w.reshape(32, 32, 3, 10)

w_min, w_max = np.min(w), np.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)
    
    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.astype('uint8'))
    plt.axis('off')
    plt.title(classes[i])
```


![output_11_0.png](https://i.loli.net/2020/02/29/KVoDu9cl7i6ZsHX.png)

-------

Link to the complete code：[https://github.com/ctttt1119/cs231n](https://github.com/ctttt1119/cs231n)