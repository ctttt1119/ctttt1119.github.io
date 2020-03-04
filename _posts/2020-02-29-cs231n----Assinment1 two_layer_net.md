---
layout:     post
title:      cs231n
subtitle:   two layer net
date:       2020-03-04
author:     Shawn
header-img: img/Stanford.jpg
catalog: true
tags:
    - cs231n














---

# Implementing a Neural Network

In this exercise we will develop a neural network with fully-connected layers to perform classification, and test it out on the CIFAR-10 dataset.


```python
# A bit of setup

import numpy as np
import matplotlib.pyplot as plt

from cs231n.classifiers.neural_net import TwoLayerNet

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
%load_ext autoreload
%autoreload 2

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
```

We will use the class `TwoLayerNet` in the file `cs231n/classifiers/neural_net.py` to represent instances of our network. The network parameters are stored in the instance variable `self.params` where keys are string parameter names and values are numpy arrays. Below, we initialize toy data and a toy model that we will use to develop your implementation.


```python
# Create a small net and some toy data to check your implementations.
# Note that we set the random seed for repeatable experiments.

input_size = 4
hidden_size = 10
num_classes = 3
num_inputs = 5

def init_toy_model():
    np.random.seed(0)
    return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)

def init_toy_data():
    np.random.seed(1)
    X = 10 * np.random.randn(num_inputs, input_size)
    y = np.array([0, 1, 2, 2, 1])
    return X, y

net = init_toy_model()
X, y = init_toy_data()
```

# Forward pass: compute scores
Open the file `cs231n/classifiers/neural_net.py` and look at the method `TwoLayerNet.loss`. This function is very similar to the loss functions you have written for the SVM and Softmax exercises: It takes the data and weights and computes the class scores, the loss, and the gradients on the parameters. 

Implement the first part of the forward pass which uses the weights and biases to compute the scores for all inputs.


```python
scores = net.loss(X)
print('Your scores:')
print(scores)
print()
print('correct scores:')
correct_scores = np.asarray([
  [-0.81233741, -1.27654624, -0.70335995],
  [-0.17129677, -1.18803311, -0.47310444],
  [-0.51590475, -1.01354314, -0.8504215 ],
  [-0.15419291, -0.48629638, -0.52901952],
  [-0.00618733, -0.12435261, -0.15226949]])
print(correct_scores)
print()

# The difference should be very small. We get < 1e-7
print('Difference between your scores and correct scores:')
print(np.sum(np.abs(scores - correct_scores)))
```

    Your scores:
    [[-0.81233741 -1.27654624 -0.70335995]
     [-0.17129677 -1.18803311 -0.47310444]
     [-0.51590475 -1.01354314 -0.8504215 ]
     [-0.15419291 -0.48629638 -0.52901952]
     [-0.00618733 -0.12435261 -0.15226949]]
    
    correct scores:
    [[-0.81233741 -1.27654624 -0.70335995]
     [-0.17129677 -1.18803311 -0.47310444]
     [-0.51590475 -1.01354314 -0.8504215 ]
     [-0.15419291 -0.48629638 -0.52901952]
     [-0.00618733 -0.12435261 -0.15226949]]
    
    Difference between your scores and correct scores:
    3.6802720496109664e-08


# Forward pass: compute loss
In the same function, implement the second part that computes the data and regularization loss.


```python
loss, _ = net.loss(X, y, reg=0.05)
correct_loss = 1.30378789133

# should be very small, we get < 1e-12
print('Difference between your loss and correct loss:')
print(np.sum(np.abs(loss - correct_loss)))
```

    Difference between your loss and correct loss:
    1.7985612998927536e-13


# Backward pass
Implement the rest of the function. This will compute the gradient of the loss with respect to the variables `W1`, `b1`, `W2`, and `b2`. Now that you (hopefully!) have a correctly implemented forward pass, you can debug your backward pass using a numeric gradient check:


```python
from cs231n.gradient_check import eval_numerical_gradient

# Use numeric gradient checking to check your implementation of the backward pass.
# If your implementation is correct, the difference between the numeric and
# analytic gradients should be less than 1e-8 for each of W1, W2, b1, and b2.

loss, grads = net.loss(X, y, reg=0.05)

# these should all be less than 1e-8 or so
for param_name in grads:
    f = lambda W: net.loss(X, y, reg=0.05)[0]
    param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
    print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))
```

    W1 max relative error: 3.561318e-09
    W2 max relative error: 3.440708e-09
    b1 max relative error: 1.555471e-09
    b2 max relative error: 3.865091e-11


# Train the network
To train the network we will use stochastic gradient descent (SGD), similar to the SVM and Softmax classifiers. Look at the function `TwoLayerNet.train` and fill in the missing sections to implement the training procedure. This should be very similar to the training procedure you used for the SVM and Softmax classifiers. You will also have to implement `TwoLayerNet.predict`, as the training process periodically performs prediction to keep track of accuracy over time while the network trains.

Once you have implemented the method, run the code below to train a two-layer network on toy data. You should achieve a training loss less than 0.02.


```python
net = init_toy_model()
stats = net.train(X, y, X, y,
            learning_rate=1e-1, reg=5e-6,
            num_iters=100, verbose=False)

print('Final training loss: ', stats['loss_history'][-1])

# plot the loss history
plt.plot(stats['loss_history'])
plt.xlabel('iteration')
plt.ylabel('training loss')
plt.title('Training Loss history')
plt.show()
```

    Final training loss:  0.017149607938732093



![output_11_1.png](https://i.loli.net/2020/03/04/nkSTfMNl9eCdbpD.png)


# Load the data
Now that you have implemented a two-layer network that passes gradient checks and works on toy data, it's time to load up our favorite CIFAR-10 data so we can use it to train a classifier on a real dataset.


```python
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
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
        
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test


# Invoke the above function to get our data.
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
```

    Train data shape:  (49000, 3072)
    Train labels shape:  (49000,)
    Validation data shape:  (1000, 3072)
    Validation labels shape:  (1000,)
    Test data shape:  (1000, 3072)
    Test labels shape:  (1000,)


# Train a network
To train our network we will use SGD. In addition, we will adjust the learning rate with an exponential learning rate schedule as optimization proceeds; after each epoch, we will reduce the learning rate by multiplying it by a decay rate.


```python
input_size = 32 * 32 * 3
hidden_size = 50
num_classes = 10
net = TwoLayerNet(input_size, hidden_size, num_classes)

# Train the network
stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=1000, batch_size=200,
            learning_rate=1e-4, learning_rate_decay=0.95,
            reg=0.25, verbose=True)

# Predict on the validation set
val_acc = (net.predict(X_val) == y_val).mean()
print('Validation accuracy: ', val_acc)

```

    iteration 0 / 1000: loss 2.302954
    iteration 100 / 1000: loss 2.302304
    iteration 200 / 1000: loss 2.294888
    iteration 300 / 1000: loss 2.250345
    iteration 400 / 1000: loss 2.179656
    iteration 500 / 1000: loss 2.130062
    iteration 600 / 1000: loss 2.132344
    iteration 700 / 1000: loss 2.122066
    iteration 800 / 1000: loss 2.001848
    iteration 900 / 1000: loss 1.962665
    Validation accuracy:  0.288


# Debug the training
With the default parameters we provided above, you should get a validation accuracy of about 0.29 on the validation set. This isn't very good.

One strategy for getting insight into what's wrong is to plot the loss function and the accuracies on the training and validation sets during optimization.

Another strategy is to visualize the weights that were learned in the first layer of the network. In most neural networks trained on visual data, the first layer weights typically show some visible structure when visualized.


```python
# Plot the loss function and train / validation accuracies
plt.subplot(2, 1, 1)
plt.plot(stats['loss_history'])
plt.title('Loss history')
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.subplot(2, 1, 2)
plt.plot(stats['train_acc_history'], label='train')
plt.plot(stats['val_acc_history'], label='val')
plt.title('Classification accuracy history')
plt.xlabel('Epoch')
plt.ylabel('Classification accuracy')
plt.legend()
plt.show()
```


![output_17_0.png](https://i.loli.net/2020/03/04/GCOzWNtZHLYcFQr.png)

```python
from cs231n.vis_utils import visualize_grid

# Visualize the weights of the network

def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()

show_net_weights(net)
```


![output_18_0.png](https://i.loli.net/2020/03/04/sOmy9SbVWFfKtXP.png)


# Tune your hyperparameters

**What's wrong?**. Looking at the visualizations above, we see that the loss is decreasing more or less linearly, which seems to suggest that the learning rate may be too low. Moreover, there is no gap between the training and validation accuracy, suggesting that the model we used has low capacity, and that we should increase its size. On the other hand, with a very large model we would expect to see more overfitting, which would manifest itself as a very large gap between the training and validation accuracy.

**Tuning**. Tuning the hyperparameters and developing intuition for how they affect the final performance is a large part of using Neural Networks, so we want you to get a lot of practice. Below, you should experiment with different values of the various hyperparameters, including hidden layer size, learning rate, numer of training epochs, and regularization strength. You might also consider tuning the learning rate decay, but you should be able to get good performance using the default value.

**Approximate results**. You should be aim to achieve a classification accuracy of greater than 48% on the validation set. Our best network gets over 52% on the validation set.

**Experiment**: You goal in this exercise is to get as good of a result on CIFAR-10 as you can (52% could serve as a reference), with a fully-connected Neural Network. Feel free implement your own techniques (e.g. PCA to reduce dimensionality, or adding dropout, or adding features to the solver, etc.).

**Explain your hyperparameter tuning process below.**

$\color{blue}{\textit Your Answer:}$


```python
best_net = None # store the best model into this 

#################################################################################
# TODO: Tune hyperparameters using the validation set. Store your best trained  #
# model in best_net.                                                            #
#                                                                               #
# To help debug your network, it may help to use visualizations similar to the  #
# ones we used above; these visualizations will have significant qualitative    #
# differences from the ones we saw above for the poorly tuned network.          #
#                                                                               #
# Tweaking hyperparameters by hand can be fun, but you might find it useful to  #
# write code to sweep through possible combinations of hyperparameters          #
# automatically like we did on the previous exercises.                          #
#################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

input_size = 32 * 32 * 3
hidden_size = [50,75,100,125,150]
num_classes = 10
reg = [0.10,0.15,0.20,0.25,0.30]
learing_rate = [1e-3]
best_acc = -0.10
results = {}

for hs in hidden_size:
    net = TwoLayerNet(input_size, hs, num_classes)
    for rs in reg:
        for lr in learing_rate:
            stats = net.train(X_train, y_train, X_val, y_val,
            num_iters=2000, batch_size=200,
            learning_rate=lr, learning_rate_decay=0.95,
            reg=rs, verbose=True)
            val_acc = (net.predict(X_val) == y_val).mean()
            results[(hs,lr,rs)] = val_acc
            if (val_acc > best_acc):
                best_acc = val_acc
                best_net = net
for hidden_size,learing_rate, reg in sorted(results):
    val_acc = results[(hs,lr,rs)]
    print('hidden_size %d learing_rate %e reg %e val_acc: %f' % (
                hs,lr, rs, val_acc))
    
print('best validation accuracy achieved during cross-validation: %f' % best_acc)


# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

```

    iteration 0 / 2000: loss 2.302730
    iteration 100 / 2000: loss 1.965422
    iteration 200 / 2000: loss 1.773385
    iteration 300 / 2000: loss 1.838617
    iteration 400 / 2000: loss 1.615767
    iteration 500 / 2000: loss 1.649003
    iteration 600 / 2000: loss 1.590548
    iteration 700 / 2000: loss 1.500459
    iteration 800 / 2000: loss 1.440769
    iteration 900 / 2000: loss 1.438617
    iteration 1000 / 2000: loss 1.429198
    iteration 1100 / 2000: loss 1.495685
    iteration 1200 / 2000: loss 1.381436
    iteration 1300 / 2000: loss 1.364247
    iteration 1400 / 2000: loss 1.386443
    iteration 1500 / 2000: loss 1.494623
    iteration 1600 / 2000: loss 1.418557
    iteration 1700 / 2000: loss 1.211028
    iteration 1800 / 2000: loss 1.333260
    iteration 1900 / 2000: loss 1.393779
    iteration 0 / 2000: loss 1.486854
    iteration 100 / 2000: loss 1.312871
    iteration 200 / 2000: loss 1.432230
    iteration 300 / 2000: loss 1.376025
    iteration 400 / 2000: loss 1.425319
    iteration 500 / 2000: loss 1.497926
    iteration 600 / 2000: loss 1.453487
    iteration 700 / 2000: loss 1.365877
    iteration 800 / 2000: loss 1.363551
    iteration 900 / 2000: loss 1.407413
    iteration 1000 / 2000: loss 1.421251
    iteration 1100 / 2000: loss 1.542119
    iteration 1200 / 2000: loss 1.200041
    iteration 1300 / 2000: loss 1.507024
    iteration 1400 / 2000: loss 1.261065
    iteration 1500 / 2000: loss 1.513868
    iteration 1600 / 2000: loss 1.303798
    iteration 1700 / 2000: loss 1.325905
    iteration 1800 / 2000: loss 1.391217
    iteration 1900 / 2000: loss 1.261005
    iteration 0 / 2000: loss 1.445467
    iteration 100 / 2000: loss 1.435386
    iteration 200 / 2000: loss 1.531935
    iteration 300 / 2000: loss 1.312147
    iteration 400 / 2000: loss 1.420455
    iteration 500 / 2000: loss 1.391463
    iteration 600 / 2000: loss 1.319927
    iteration 700 / 2000: loss 1.389912
    iteration 800 / 2000: loss 1.376427
    iteration 900 / 2000: loss 1.315287
    iteration 1000 / 2000: loss 1.342393
    iteration 1100 / 2000: loss 1.391542
    iteration 1200 / 2000: loss 1.544231
    iteration 1300 / 2000: loss 1.400330
    iteration 1400 / 2000: loss 1.311174
    iteration 1500 / 2000: loss 1.317460
    iteration 1600 / 2000: loss 1.367942
    iteration 1700 / 2000: loss 1.449592
    iteration 1800 / 2000: loss 1.359531
    iteration 1900 / 2000: loss 1.496698
    iteration 0 / 2000: loss 1.457439
    iteration 100 / 2000: loss 1.620133
    iteration 200 / 2000: loss 1.611454
    iteration 300 / 2000: loss 1.389349
    iteration 400 / 2000: loss 1.477923
    iteration 500 / 2000: loss 1.434338
    iteration 600 / 2000: loss 1.392571
    iteration 700 / 2000: loss 1.304812
    iteration 800 / 2000: loss 1.402552
    iteration 900 / 2000: loss 1.288260
    iteration 1000 / 2000: loss 1.425948
    iteration 1100 / 2000: loss 1.318647
    iteration 1200 / 2000: loss 1.363053
    iteration 1300 / 2000: loss 1.455710
    iteration 1400 / 2000: loss 1.447480
    iteration 1500 / 2000: loss 1.408263
    iteration 1600 / 2000: loss 1.448944
    iteration 1700 / 2000: loss 1.350658
    iteration 1800 / 2000: loss 1.431767
    iteration 1900 / 2000: loss 1.405079
    iteration 0 / 2000: loss 1.315082
    iteration 100 / 2000: loss 1.419048
    iteration 200 / 2000: loss 1.574873
    iteration 300 / 2000: loss 1.473231
    iteration 400 / 2000: loss 1.391615
    iteration 500 / 2000: loss 1.446914
    iteration 600 / 2000: loss 1.448870
    iteration 700 / 2000: loss 1.457617
    iteration 800 / 2000: loss 1.330638
    iteration 900 / 2000: loss 1.329814
    iteration 1000 / 2000: loss 1.445767
    iteration 1100 / 2000: loss 1.415865
    iteration 1200 / 2000: loss 1.381459
    iteration 1300 / 2000: loss 1.335687
    iteration 1400 / 2000: loss 1.264320
    iteration 1500 / 2000: loss 1.447820
    iteration 1600 / 2000: loss 1.497468
    iteration 1700 / 2000: loss 1.507701
    iteration 1800 / 2000: loss 1.430715
    iteration 1900 / 2000: loss 1.412659
    iteration 0 / 2000: loss 2.302827
    iteration 100 / 2000: loss 1.980791
    iteration 200 / 2000: loss 1.638476
    iteration 300 / 2000: loss 1.634694
    iteration 400 / 2000: loss 1.434348
    iteration 500 / 2000: loss 1.645159
    iteration 600 / 2000: loss 1.594838
    iteration 700 / 2000: loss 1.515251
    iteration 800 / 2000: loss 1.518781
    iteration 900 / 2000: loss 1.509252
    iteration 1000 / 2000: loss 1.491334
    iteration 1100 / 2000: loss 1.569436
    iteration 1200 / 2000: loss 1.404996
    iteration 1300 / 2000: loss 1.315910
    iteration 1400 / 2000: loss 1.340791
    iteration 1500 / 2000: loss 1.536333
    iteration 1600 / 2000: loss 1.327953
    iteration 1700 / 2000: loss 1.432468
    iteration 1800 / 2000: loss 1.288216
    iteration 1900 / 2000: loss 1.257117
    iteration 0 / 2000: loss 1.344520
    iteration 100 / 2000: loss 1.353493
    iteration 200 / 2000: loss 1.353886
    iteration 300 / 2000: loss 1.423691
    iteration 400 / 2000: loss 1.380827
    iteration 500 / 2000: loss 1.398219
    iteration 600 / 2000: loss 1.298914
    iteration 700 / 2000: loss 1.467094
    iteration 800 / 2000: loss 1.258707
    iteration 900 / 2000: loss 1.521835
    iteration 1000 / 2000: loss 1.336051
    iteration 1100 / 2000: loss 1.317035
    iteration 1200 / 2000: loss 1.292332
    iteration 1300 / 2000: loss 1.269856
    iteration 1400 / 2000: loss 1.254234
    iteration 1500 / 2000: loss 1.305914
    iteration 1600 / 2000: loss 1.444975
    iteration 1700 / 2000: loss 1.282062
    iteration 1800 / 2000: loss 1.225869
    iteration 1900 / 2000: loss 1.309065
    iteration 0 / 2000: loss 1.357122
    iteration 100 / 2000: loss 1.524630
    iteration 200 / 2000: loss 1.368655
    iteration 300 / 2000: loss 1.610314
    iteration 400 / 2000: loss 1.570451
    iteration 500 / 2000: loss 1.432638
    iteration 600 / 2000: loss 1.492545
    iteration 700 / 2000: loss 1.292072
    iteration 800 / 2000: loss 1.303469
    iteration 900 / 2000: loss 1.276486
    iteration 1000 / 2000: loss 1.327985
    iteration 1100 / 2000: loss 1.489971
    iteration 1200 / 2000: loss 1.427303
    iteration 1300 / 2000: loss 1.340195
    iteration 1400 / 2000: loss 1.396135
    iteration 1500 / 2000: loss 1.276258
    iteration 1600 / 2000: loss 1.409236
    iteration 1700 / 2000: loss 1.416500
    iteration 1800 / 2000: loss 1.303523
    iteration 1900 / 2000: loss 1.344292
    iteration 0 / 2000: loss 1.330885
    iteration 100 / 2000: loss 1.643684
    iteration 200 / 2000: loss 1.567131
    iteration 300 / 2000: loss 1.374515
    iteration 400 / 2000: loss 1.452344
    iteration 500 / 2000: loss 1.452339
    iteration 600 / 2000: loss 1.346401
    iteration 700 / 2000: loss 1.401640
    iteration 800 / 2000: loss 1.494398
    iteration 900 / 2000: loss 1.331481
    iteration 1000 / 2000: loss 1.415521
    iteration 1100 / 2000: loss 1.370305
    iteration 1200 / 2000: loss 1.432291
    iteration 1300 / 2000: loss 1.448133
    iteration 1400 / 2000: loss 1.318600
    iteration 1500 / 2000: loss 1.405143
    iteration 1600 / 2000: loss 1.441901
    iteration 1700 / 2000: loss 1.343032
    iteration 1800 / 2000: loss 1.432415
    iteration 1900 / 2000: loss 1.333095
    iteration 0 / 2000: loss 1.383340
    iteration 100 / 2000: loss 1.509921
    iteration 200 / 2000: loss 1.457432
    iteration 300 / 2000: loss 1.372504
    iteration 400 / 2000: loss 1.520937
    iteration 500 / 2000: loss 1.397279
    iteration 600 / 2000: loss 1.372101
    iteration 700 / 2000: loss 1.425149
    iteration 800 / 2000: loss 1.503649
    iteration 900 / 2000: loss 1.420579
    iteration 1000 / 2000: loss 1.476135
    iteration 1100 / 2000: loss 1.406074
    iteration 1200 / 2000: loss 1.414907
    iteration 1300 / 2000: loss 1.502064
    iteration 1400 / 2000: loss 1.282453
    iteration 1500 / 2000: loss 1.310215
    iteration 1600 / 2000: loss 1.436106
    iteration 1700 / 2000: loss 1.509194
    iteration 1800 / 2000: loss 1.428124
    iteration 1900 / 2000: loss 1.446370
    iteration 0 / 2000: loss 2.302867
    iteration 100 / 2000: loss 1.991596
    iteration 200 / 2000: loss 1.789323
    iteration 300 / 2000: loss 1.591355
    iteration 400 / 2000: loss 1.558739
    iteration 500 / 2000: loss 1.500145
    iteration 600 / 2000: loss 1.405329
    iteration 700 / 2000: loss 1.422507
    iteration 800 / 2000: loss 1.516567
    iteration 900 / 2000: loss 1.505997
    iteration 1000 / 2000: loss 1.529560
    iteration 1100 / 2000: loss 1.451155
    iteration 1200 / 2000: loss 1.415207
    iteration 1300 / 2000: loss 1.364190
    iteration 1400 / 2000: loss 1.417759
    iteration 1500 / 2000: loss 1.301338
    iteration 1600 / 2000: loss 1.299214
    iteration 1700 / 2000: loss 1.505696
    iteration 1800 / 2000: loss 1.182786
    iteration 1900 / 2000: loss 1.276273
    iteration 0 / 2000: loss 1.283366
    iteration 100 / 2000: loss 1.325976
    iteration 200 / 2000: loss 1.349928
    iteration 300 / 2000: loss 1.526428
    iteration 400 / 2000: loss 1.356031
    iteration 500 / 2000: loss 1.394660
    iteration 600 / 2000: loss 1.341804
    iteration 700 / 2000: loss 1.344548
    iteration 800 / 2000: loss 1.367854
    iteration 900 / 2000: loss 1.428839
    iteration 1000 / 2000: loss 1.363419
    iteration 1100 / 2000: loss 1.290580
    iteration 1200 / 2000: loss 1.336347
    iteration 1300 / 2000: loss 1.416293
    iteration 1400 / 2000: loss 1.307705
    iteration 1500 / 2000: loss 1.305436
    iteration 1600 / 2000: loss 1.319626
    iteration 1700 / 2000: loss 1.456641
    iteration 1800 / 2000: loss 1.351797
    iteration 1900 / 2000: loss 1.402679
    iteration 0 / 2000: loss 1.265200
    iteration 100 / 2000: loss 1.385626
    iteration 200 / 2000: loss 1.352542
    iteration 300 / 2000: loss 1.228651
    iteration 400 / 2000: loss 1.417462
    iteration 500 / 2000: loss 1.383002
    iteration 600 / 2000: loss 1.299990
    iteration 700 / 2000: loss 1.284533
    iteration 800 / 2000: loss 1.301075
    iteration 900 / 2000: loss 1.306537
    iteration 1000 / 2000: loss 1.378653
    iteration 1100 / 2000: loss 1.448946
    iteration 1200 / 2000: loss 1.363098
    iteration 1300 / 2000: loss 1.241718
    iteration 1400 / 2000: loss 1.243636
    iteration 1500 / 2000: loss 1.510953
    iteration 1600 / 2000: loss 1.217033
    iteration 1700 / 2000: loss 1.408389
    iteration 1800 / 2000: loss 1.386015
    iteration 1900 / 2000: loss 1.265006
    iteration 0 / 2000: loss 1.461333
    iteration 100 / 2000: loss 1.316823
    iteration 200 / 2000: loss 1.514019
    iteration 300 / 2000: loss 1.260550
    iteration 400 / 2000: loss 1.368193
    iteration 500 / 2000: loss 1.319381
    iteration 600 / 2000: loss 1.347807
    iteration 700 / 2000: loss 1.334526
    iteration 800 / 2000: loss 1.306425
    iteration 900 / 2000: loss 1.371283
    iteration 1000 / 2000: loss 1.265303
    iteration 1100 / 2000: loss 1.388358
    iteration 1200 / 2000: loss 1.369075
    iteration 1300 / 2000: loss 1.338335
    iteration 1400 / 2000: loss 1.440725
    iteration 1500 / 2000: loss 1.386760
    iteration 1600 / 2000: loss 1.308997
    iteration 1700 / 2000: loss 1.266036
    iteration 1800 / 2000: loss 1.253737
    iteration 1900 / 2000: loss 1.231973
    iteration 0 / 2000: loss 1.358019
    iteration 100 / 2000: loss 1.446472
    iteration 200 / 2000: loss 1.528874
    iteration 300 / 2000: loss 1.321769
    iteration 400 / 2000: loss 1.502590
    iteration 500 / 2000: loss 1.312277
    iteration 600 / 2000: loss 1.379574
    iteration 700 / 2000: loss 1.375439
    iteration 800 / 2000: loss 1.424428
    iteration 900 / 2000: loss 1.461011
    iteration 1000 / 2000: loss 1.445314
    iteration 1100 / 2000: loss 1.540250
    iteration 1200 / 2000: loss 1.322529
    iteration 1300 / 2000: loss 1.306630
    iteration 1400 / 2000: loss 1.529835
    iteration 1500 / 2000: loss 1.335999
    iteration 1600 / 2000: loss 1.453812
    iteration 1700 / 2000: loss 1.539706
    iteration 1800 / 2000: loss 1.348862
    iteration 1900 / 2000: loss 1.243765
    iteration 0 / 2000: loss 2.302990
    iteration 100 / 2000: loss 1.919073
    iteration 200 / 2000: loss 1.857283
    iteration 300 / 2000: loss 1.659790
    iteration 400 / 2000: loss 1.577684
    iteration 500 / 2000: loss 1.609770
    iteration 600 / 2000: loss 1.488692
    iteration 700 / 2000: loss 1.506294
    iteration 800 / 2000: loss 1.516108
    iteration 900 / 2000: loss 1.443440
    iteration 1000 / 2000: loss 1.339864
    iteration 1100 / 2000: loss 1.363079
    iteration 1200 / 2000: loss 1.452893
    iteration 1300 / 2000: loss 1.540855
    iteration 1400 / 2000: loss 1.310576
    iteration 1500 / 2000: loss 1.456290
    iteration 1600 / 2000: loss 1.372187
    iteration 1700 / 2000: loss 1.314764
    iteration 1800 / 2000: loss 1.372391
    iteration 1900 / 2000: loss 1.385037
    iteration 0 / 2000: loss 1.230716
    iteration 100 / 2000: loss 1.646253
    iteration 200 / 2000: loss 1.433111
    iteration 300 / 2000: loss 1.327098
    iteration 400 / 2000: loss 1.472308
    iteration 500 / 2000: loss 1.284075
    iteration 600 / 2000: loss 1.276971
    iteration 700 / 2000: loss 1.401248
    iteration 800 / 2000: loss 1.414617
    iteration 900 / 2000: loss 1.292875
    iteration 1000 / 2000: loss 1.298872
    iteration 1100 / 2000: loss 1.288529
    iteration 1200 / 2000: loss 1.275230
    iteration 1300 / 2000: loss 1.253890
    iteration 1400 / 2000: loss 1.282842
    iteration 1500 / 2000: loss 1.109552
    iteration 1600 / 2000: loss 1.274977
    iteration 1700 / 2000: loss 1.279155
    iteration 1800 / 2000: loss 1.343209
    iteration 1900 / 2000: loss 1.286581
    iteration 0 / 2000: loss 1.187853
    iteration 100 / 2000: loss 1.307984
    iteration 200 / 2000: loss 1.275551
    iteration 300 / 2000: loss 1.297438
    iteration 400 / 2000: loss 1.425483
    iteration 500 / 2000: loss 1.401594
    iteration 600 / 2000: loss 1.357709
    iteration 700 / 2000: loss 1.283651
    iteration 800 / 2000: loss 1.498002
    iteration 900 / 2000: loss 1.308027
    iteration 1000 / 2000: loss 1.306133
    iteration 1100 / 2000: loss 1.462577
    iteration 1200 / 2000: loss 1.290987
    iteration 1300 / 2000: loss 1.281072
    iteration 1400 / 2000: loss 1.243936
    iteration 1500 / 2000: loss 1.263739
    iteration 1600 / 2000: loss 1.276075
    iteration 1700 / 2000: loss 1.309141
    iteration 1800 / 2000: loss 1.311491
    iteration 1900 / 2000: loss 1.136036
    iteration 0 / 2000: loss 1.267755
    iteration 100 / 2000: loss 2.156713
    iteration 200 / 2000: loss 1.311532
    iteration 300 / 2000: loss 1.393519
    iteration 400 / 2000: loss 1.369042
    iteration 500 / 2000: loss 1.385705
    iteration 600 / 2000: loss 1.338638
    iteration 700 / 2000: loss 1.395322
    iteration 800 / 2000: loss 1.374710
    iteration 900 / 2000: loss 1.367388
    iteration 1000 / 2000: loss 1.276865
    iteration 1100 / 2000: loss 1.357720
    iteration 1200 / 2000: loss 1.192257
    iteration 1300 / 2000: loss 1.359301
    iteration 1400 / 2000: loss 1.212232
    iteration 1500 / 2000: loss 1.394791
    iteration 1600 / 2000: loss 1.292686
    iteration 1700 / 2000: loss 1.300881
    iteration 1800 / 2000: loss 1.350201
    iteration 1900 / 2000: loss 1.399769
    iteration 0 / 2000: loss 1.498620
    iteration 100 / 2000: loss 1.441481
    iteration 200 / 2000: loss 1.451436
    iteration 300 / 2000: loss 1.422618
    iteration 400 / 2000: loss 1.444173
    iteration 500 / 2000: loss 1.311272
    iteration 600 / 2000: loss 1.351442
    iteration 700 / 2000: loss 1.397236
    iteration 800 / 2000: loss 1.310206
    iteration 900 / 2000: loss 1.298357
    iteration 1000 / 2000: loss 1.329000
    iteration 1100 / 2000: loss 1.309043
    iteration 1200 / 2000: loss 1.275519
    iteration 1300 / 2000: loss 1.371544
    iteration 1400 / 2000: loss 1.421344
    iteration 1500 / 2000: loss 1.415199
    iteration 1600 / 2000: loss 1.296853
    iteration 1700 / 2000: loss 1.273343
    iteration 1800 / 2000: loss 1.253117
    iteration 1900 / 2000: loss 1.401261
    iteration 0 / 2000: loss 2.303076
    iteration 100 / 2000: loss 1.951049
    iteration 200 / 2000: loss 1.755271
    iteration 300 / 2000: loss 1.689766
    iteration 400 / 2000: loss 1.580482
    iteration 500 / 2000: loss 1.555512
    iteration 600 / 2000: loss 1.617595
    iteration 700 / 2000: loss 1.593322
    iteration 800 / 2000: loss 1.522256
    iteration 900 / 2000: loss 1.461625
    iteration 1000 / 2000: loss 1.453568
    iteration 1100 / 2000: loss 1.429689
    iteration 1200 / 2000: loss 1.553202
    iteration 1300 / 2000: loss 1.479351
    iteration 1400 / 2000: loss 1.333787
    iteration 1500 / 2000: loss 1.453384
    iteration 1600 / 2000: loss 1.162057
    iteration 1700 / 2000: loss 1.282875
    iteration 1800 / 2000: loss 1.369310
    iteration 1900 / 2000: loss 1.274355
    iteration 0 / 2000: loss 1.314226
    iteration 100 / 2000: loss 1.298173
    iteration 200 / 2000: loss 1.276065
    iteration 300 / 2000: loss 1.256328
    iteration 400 / 2000: loss 1.428530
    iteration 500 / 2000: loss 1.186812
    iteration 600 / 2000: loss 1.327463
    iteration 700 / 2000: loss 1.313199
    iteration 800 / 2000: loss 1.296363
    iteration 900 / 2000: loss 1.280981
    iteration 1000 / 2000: loss 1.087332
    iteration 1100 / 2000: loss 1.224225
    iteration 1200 / 2000: loss 1.353654
    iteration 1300 / 2000: loss 1.284497
    iteration 1400 / 2000: loss 1.287487
    iteration 1500 / 2000: loss 1.373481
    iteration 1600 / 2000: loss 1.334620
    iteration 1700 / 2000: loss 1.273974
    iteration 1800 / 2000: loss 1.163242
    iteration 1900 / 2000: loss 1.219571
    iteration 0 / 2000: loss 1.218861
    iteration 100 / 2000: loss 1.440996
    iteration 200 / 2000: loss 1.425708
    iteration 300 / 2000: loss 1.274889
    iteration 400 / 2000: loss 1.361275
    iteration 500 / 2000: loss 1.250173
    iteration 600 / 2000: loss 1.238631
    iteration 700 / 2000: loss 1.181168
    iteration 800 / 2000: loss 1.304190
    iteration 900 / 2000: loss 1.239623
    iteration 1000 / 2000: loss 1.225788
    iteration 1100 / 2000: loss 1.261434
    iteration 1200 / 2000: loss 1.217082
    iteration 1300 / 2000: loss 1.308132
    iteration 1400 / 2000: loss 1.220070
    iteration 1500 / 2000: loss 1.226443
    iteration 1600 / 2000: loss 1.296900
    iteration 1700 / 2000: loss 1.227241
    iteration 1800 / 2000: loss 1.179182
    iteration 1900 / 2000: loss 1.174472
    iteration 0 / 2000: loss 1.306679
    iteration 100 / 2000: loss 1.343482
    iteration 200 / 2000: loss 1.406097
    iteration 300 / 2000: loss 1.467929
    iteration 400 / 2000: loss 1.312472
    iteration 500 / 2000: loss 1.322241
    iteration 600 / 2000: loss 1.288072
    iteration 700 / 2000: loss 1.300123
    iteration 800 / 2000: loss 1.453212
    iteration 900 / 2000: loss 1.423402
    iteration 1000 / 2000: loss 1.127594
    iteration 1100 / 2000: loss 1.190205
    iteration 1200 / 2000: loss 1.229713
    iteration 1300 / 2000: loss 1.259612
    iteration 1400 / 2000: loss 1.258211
    iteration 1500 / 2000: loss 1.366010
    iteration 1600 / 2000: loss 1.467970
    iteration 1700 / 2000: loss 1.472616
    iteration 1800 / 2000: loss 1.309318
    iteration 1900 / 2000: loss 1.295704
    iteration 0 / 2000: loss 1.301666
    iteration 100 / 2000: loss 1.523744
    iteration 200 / 2000: loss 1.497299
    iteration 300 / 2000: loss 1.364337
    iteration 400 / 2000: loss 1.389935
    iteration 500 / 2000: loss 1.373208
    iteration 600 / 2000: loss 1.350623
    iteration 700 / 2000: loss 1.441617
    iteration 800 / 2000: loss 1.409233
    iteration 900 / 2000: loss 1.382814
    iteration 1000 / 2000: loss 1.336699
    iteration 1100 / 2000: loss 1.409010
    iteration 1200 / 2000: loss 1.412737
    iteration 1300 / 2000: loss 1.195499
    iteration 1400 / 2000: loss 1.431553
    iteration 1500 / 2000: loss 1.327145
    iteration 1600 / 2000: loss 1.336456
    iteration 1700 / 2000: loss 1.244778
    iteration 1800 / 2000: loss 1.378489
    iteration 1900 / 2000: loss 1.246795
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    hidden_size 150 learing_rate 1.000000e-03 reg 3.000000e-01 val_acc: 0.530000
    best validation accuracy achieved during cross-validation: 0.547000



```python
# visualize the weights of the best network
show_net_weights(best_net)
```


![output_22_0.png](https://i.loli.net/2020/03/04/TeLJ7Xp1zDxi3Zq.png)


# Run on the test set
When you are done experimenting, you should evaluate your final trained network on the test set; you should get above 48%.


```python
test_acc = (best_net.predict(X_test) == y_test).mean()
print('Test accuracy: ', test_acc)
```

    Test accuracy:  0.54


**Inline Question**

Now that you have trained a Neural Network classifier, you may find that your testing accuracy is much lower than the training accuracy. In what ways can we decrease this gap? Select all that apply.

1. Train on a larger dataset.
2. Add more hidden units.
3. Increase the regularization strength.
4. None of the above.

Your Answer:

Your Explanation:

----

Link to the complete code：[https://github.com/ctttt1119/cs231n](https://github.com/ctttt1119/cs231n)


