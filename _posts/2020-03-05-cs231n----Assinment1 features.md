---
layout:     post
title:      cs231n
subtitle:   Image features exercise
date:       2020-03-05
author:     Shawn
header-img: img/Stanford.jpg
catalog: true
tags:
    - cs231n















---

# Image features exercise

*Complete and hand in this completed worksheet (including its outputs and any supporting code outside of the worksheet) with your assignment submission. For more details see the [assignments page](http://vision.stanford.edu/teaching/cs231n/assignments.html) on the course website.*

We have seen that we can achieve reasonable performance on an image classification task by training a linear classifier on the pixels of the input image. In this exercise we will show that we can improve our classification performance by training linear classifiers not on raw pixels but on features that are computed from the raw pixels.

All of your work for this exercise will be done in this notebook.


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

## Load data
Similar to previous exercises, we will load CIFAR-10 data from disk.


```python
from cs231n.features import color_histogram_hsv, hog_feature

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
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
    
    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()
```

## Extract Features
For each image we will compute a Histogram of Oriented
Gradients (HOG) as well as a color histogram using the hue channel in HSV
color space. We form our final feature vector for each image by concatenating
the HOG and color histogram feature vectors.

Roughly speaking, HOG should capture the texture of the image while ignoring
color information, and the color histogram represents the color of the input
image while ignoring texture. As a result, we expect that using both together
ought to work better than using either alone. Verifying this assumption would
be a good thing to try for your own interest.

The `hog_feature` and `color_histogram_hsv` functions both operate on a single
image and return a feature vector for that image. The extract_features
function takes a set of images and a list of feature functions and evaluates
each feature function on each image, storing the results in a matrix where
each column is the concatenation of all feature vectors for a single image.


```python
from cs231n.features import *

num_color_bins = 10 # Number of bins in the color histogram
feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
X_train_feats = extract_features(X_train, feature_fns, verbose=True)
X_val_feats = extract_features(X_val, feature_fns)
X_test_feats = extract_features(X_test, feature_fns)

# Preprocessing: Subtract the mean feature
mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
X_train_feats -= mean_feat
X_val_feats -= mean_feat
X_test_feats -= mean_feat

# Preprocessing: Divide by standard deviation. This ensures that each feature
# has roughly the same scale.
std_feat = np.std(X_train_feats, axis=0, keepdims=True)
X_train_feats /= std_feat
X_val_feats /= std_feat
X_test_feats /= std_feat

# Preprocessing: Add a bias dimension
X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])
```

    Done extracting features for 1000 / 49000 images
    Done extracting features for 2000 / 49000 images
    Done extracting features for 3000 / 49000 images
    Done extracting features for 4000 / 49000 images
    Done extracting features for 5000 / 49000 images
    Done extracting features for 6000 / 49000 images
    Done extracting features for 7000 / 49000 images
    Done extracting features for 8000 / 49000 images
    Done extracting features for 9000 / 49000 images
    Done extracting features for 10000 / 49000 images
    Done extracting features for 11000 / 49000 images
    Done extracting features for 12000 / 49000 images
    Done extracting features for 13000 / 49000 images
    Done extracting features for 14000 / 49000 images
    Done extracting features for 15000 / 49000 images
    Done extracting features for 16000 / 49000 images
    Done extracting features for 17000 / 49000 images
    Done extracting features for 18000 / 49000 images
    Done extracting features for 19000 / 49000 images
    Done extracting features for 20000 / 49000 images
    Done extracting features for 21000 / 49000 images
    Done extracting features for 22000 / 49000 images
    Done extracting features for 23000 / 49000 images
    Done extracting features for 24000 / 49000 images
    Done extracting features for 25000 / 49000 images
    Done extracting features for 26000 / 49000 images
    Done extracting features for 27000 / 49000 images
    Done extracting features for 28000 / 49000 images
    Done extracting features for 29000 / 49000 images
    Done extracting features for 30000 / 49000 images
    Done extracting features for 31000 / 49000 images
    Done extracting features for 32000 / 49000 images
    Done extracting features for 33000 / 49000 images
    Done extracting features for 34000 / 49000 images
    Done extracting features for 35000 / 49000 images
    Done extracting features for 36000 / 49000 images
    Done extracting features for 37000 / 49000 images
    Done extracting features for 38000 / 49000 images
    Done extracting features for 39000 / 49000 images
    Done extracting features for 40000 / 49000 images
    Done extracting features for 41000 / 49000 images
    Done extracting features for 42000 / 49000 images
    Done extracting features for 43000 / 49000 images
    Done extracting features for 44000 / 49000 images
    Done extracting features for 45000 / 49000 images
    Done extracting features for 46000 / 49000 images
    Done extracting features for 47000 / 49000 images
    Done extracting features for 48000 / 49000 images
    Done extracting features for 49000 / 49000 images


## Train SVM on features
Using the multiclass SVM code developed earlier in the assignment, train SVMs on top of the features extracted above; this should achieve better results than training SVMs directly on top of raw pixels.


```python
# Use the validation set to tune the learning rate and regularization strength

from cs231n.classifiers.linear_classifier import LinearSVM

learning_rates = [1e-9, 1e-8, 1e-7]
regularization_strengths = [5e4, 5e5, 5e6]

results = {}
best_val = -1
best_svm = None

################################################################################
# TODO:                                                                        #
# Use the validation set to set the learning rate and regularization strength. #
# This should be identical to the validation that you did for the SVM; save    #
# the best trained classifer in best_svm. You might also want to play          #
# with different numbers of bins in the color histogram. If you are careful    #
# you should be able to get accuracy of near 0.44 on the validation set.       #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

for lr in learning_rates:
    for reg in regularization_strengths:
        svm = LinearSVM()
        loss_history = svm.train(X_train_feats, y_train, learning_rate=lr, reg=reg, num_iters=2000)
        y_train_pred = svm.predict(X_train_feats)
        y_val_pred = svm.predict(X_val_feats)
        train_accuracy = np.mean(y_train == y_train_pred)
        val_accuracy = np.mean(y_val == y_val_pred)
        results[(lr,reg)] = [train_accuracy, val_accuracy]
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
                lr, reg, train_accuracy, val_accuracy))
    
print('best validation accuracy achieved during cross-validation: %f' % best_val)
```

    lr 1.000000e-09 reg 5.000000e+04 train accuracy: 0.093367 val accuracy: 0.081000
    lr 1.000000e-09 reg 5.000000e+05 train accuracy: 0.076347 val accuracy: 0.078000
    lr 1.000000e-09 reg 5.000000e+06 train accuracy: 0.412429 val accuracy: 0.416000
    lr 1.000000e-08 reg 5.000000e+04 train accuracy: 0.100204 val accuracy: 0.099000
    lr 1.000000e-08 reg 5.000000e+05 train accuracy: 0.413612 val accuracy: 0.419000
    lr 1.000000e-08 reg 5.000000e+06 train accuracy: 0.408061 val accuracy: 0.400000
    lr 1.000000e-07 reg 5.000000e+04 train accuracy: 0.412592 val accuracy: 0.422000
    lr 1.000000e-07 reg 5.000000e+05 train accuracy: 0.396571 val accuracy: 0.393000
    lr 1.000000e-07 reg 5.000000e+06 train accuracy: 0.330082 val accuracy: 0.348000
    best validation accuracy achieved during cross-validation: 0.422000



```python
# Evaluate your trained SVM on the test set
y_test_pred = best_svm.predict(X_test_feats)
test_accuracy = np.mean(y_test == y_test_pred)
print(test_accuracy)
```

    0.422



```python
# An important way to gain intuition about how an algorithm works is to
# visualize the mistakes that it makes. In this visualization, we show examples
# of images that are misclassified by our current system. The first column
# shows images that our system labeled as "plane" but whose true label is
# something other than "plane".

examples_per_class = 8
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for cls, cls_name in enumerate(classes):
    idxs = np.where((y_test != cls) & (y_test_pred == cls))[0]
    idxs = np.random.choice(idxs, examples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt.subplot(examples_per_class, len(classes), i * len(classes) + cls + 1)
        plt.imshow(X_test[idx].astype('uint8'))
        plt.axis('off')
        if i == 0:
            plt.title(cls_name)
plt.show()
```


![output_9_0.png](https://i.loli.net/2020/03/04/HcRkdrL6o5pijl3.png)


### Inline question 1:
Describe the misclassification results that you see. Do they make sense?


Your Answer:




## Neural Network on image features
Earlier in this assigment we saw that training a two-layer neural network on raw pixels achieved better classification performance than linear classifiers on raw pixels. In this notebook we have seen that linear classifiers on image features outperform linear classifiers on raw pixels. 

For completeness, we should also try training a neural network on image features. This approach should outperform all previous approaches: you should easily be able to achieve over 55% classification accuracy on the test set; our best model achieves about 60% classification accuracy.


```python
# Preprocessing: Remove the bias dimension
# Make sure to run this cell only ONCE
print(X_train_feats.shape)
X_train_feats = X_train_feats[:, :-1]
X_val_feats = X_val_feats[:, :-1]
X_test_feats = X_test_feats[:, :-1]

print(X_train_feats.shape)
```

    (49000, 155)
    (49000, 154)



```python
from cs231n.classifiers.neural_net import TwoLayerNet

input_dim = X_train_feats.shape[1]
hidden_dim = 500
num_classes = 10

net = TwoLayerNet(input_dim, hidden_dim, num_classes)
best_net = None


################################################################################
# TODO: Train a two-layer neural network on image features. You may want to    #
# cross-validate various parameters as in previous sections. Store your best   #
# model in the best_net variable.                                              #
################################################################################
# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

results = {}
learning_rates = [0.25,0.30,0.35]
regularization_strengths = [1e-3,3e-3,5e-3]
best_acc = -0.1
net = TwoLayerNet(input_dim, hidden_dim, num_classes)
for lr in learning_rates:
    for rs in regularization_strengths:
        stats = net.train(X_train_feats, y_train, X_val_feats, y_val,
        num_iters=2000, batch_size=200,
        learning_rate=lr, learning_rate_decay=0.95,
        reg=rs, verbose=True)
        val_acc = (net.predict(X_val_feats) == y_val).mean()
        results[(lr,rs)] = val_acc
        if (val_acc > best_acc):
            best_acc = val_acc
            best_net = net
for lr, rs in sorted(results):
    val_acc = results[(lr,rs)]
    print('learing_rate %e reg %e val_acc: %f' % (
                lr, rs, val_acc))
    
print('best validation accuracy achieved during cross-validation: %f' % best_acc)

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

```

    iteration 0 / 2000: loss 2.302586
    iteration 100 / 2000: loss 1.956847
    iteration 200 / 2000: loss 1.706290
    iteration 300 / 2000: loss 1.566228
    iteration 400 / 2000: loss 1.551093
    iteration 500 / 2000: loss 1.327312
    iteration 600 / 2000: loss 1.284370
    iteration 700 / 2000: loss 1.386756
    iteration 800 / 2000: loss 1.192524
    iteration 900 / 2000: loss 1.362784
    iteration 1000 / 2000: loss 1.333314
    iteration 1100 / 2000: loss 1.148005
    iteration 1200 / 2000: loss 1.233245
    iteration 1300 / 2000: loss 1.067286
    iteration 1400 / 2000: loss 1.275794
    iteration 1500 / 2000: loss 1.176706
    iteration 1600 / 2000: loss 1.050405
    iteration 1700 / 2000: loss 1.318247
    iteration 1800 / 2000: loss 1.206095
    iteration 1900 / 2000: loss 1.054556
    iteration 0 / 2000: loss 1.433337
    iteration 100 / 2000: loss 1.376698
    iteration 200 / 2000: loss 1.358294
    iteration 300 / 2000: loss 1.228717
    iteration 400 / 2000: loss 1.340029
    iteration 500 / 2000: loss 1.261271
    iteration 600 / 2000: loss 1.406385
    iteration 700 / 2000: loss 1.397775
    iteration 800 / 2000: loss 1.267003
    iteration 900 / 2000: loss 1.270066
    iteration 1000 / 2000: loss 1.367221
    iteration 1100 / 2000: loss 1.270790
    iteration 1200 / 2000: loss 1.371811
    iteration 1300 / 2000: loss 1.402023
    iteration 1400 / 2000: loss 1.308130
    iteration 1500 / 2000: loss 1.200409
    iteration 1600 / 2000: loss 1.153022
    iteration 1700 / 2000: loss 1.305232
    iteration 1800 / 2000: loss 1.270122
    iteration 1900 / 2000: loss 1.310073
    iteration 0 / 2000: loss 1.419880
    iteration 100 / 2000: loss 1.432880
    iteration 200 / 2000: loss 1.482756
    iteration 300 / 2000: loss 1.473420
    iteration 400 / 2000: loss 1.353398
    iteration 500 / 2000: loss 1.454216
    iteration 600 / 2000: loss 1.436627
    iteration 700 / 2000: loss 1.621099
    iteration 800 / 2000: loss 1.494785
    iteration 900 / 2000: loss 1.414166
    iteration 1000 / 2000: loss 1.548151
    iteration 1100 / 2000: loss 1.373051
    iteration 1200 / 2000: loss 1.453555
    iteration 1300 / 2000: loss 1.408975
    iteration 1400 / 2000: loss 1.521160
    iteration 1500 / 2000: loss 1.421060
    iteration 1600 / 2000: loss 1.402773
    iteration 1700 / 2000: loss 1.484677
    iteration 1800 / 2000: loss 1.496708
    iteration 1900 / 2000: loss 1.491577
    iteration 0 / 2000: loss 1.183833
    iteration 100 / 2000: loss 1.096780
    iteration 200 / 2000: loss 1.138768
    iteration 300 / 2000: loss 1.340814
    iteration 400 / 2000: loss 1.090285
    iteration 500 / 2000: loss 1.241176
    iteration 600 / 2000: loss 1.286156
    iteration 700 / 2000: loss 1.018906
    iteration 800 / 2000: loss 1.096595
    iteration 900 / 2000: loss 1.154339
    iteration 1000 / 2000: loss 1.010451
    iteration 1100 / 2000: loss 1.142424
    iteration 1200 / 2000: loss 1.032650
    iteration 1300 / 2000: loss 1.029679
    iteration 1400 / 2000: loss 1.120763
    iteration 1500 / 2000: loss 1.089975
    iteration 1600 / 2000: loss 1.076761
    iteration 1700 / 2000: loss 1.108918
    iteration 1800 / 2000: loss 1.076687
    iteration 1900 / 2000: loss 0.988502
    iteration 0 / 2000: loss 1.488478
    iteration 100 / 2000: loss 1.348906
    iteration 200 / 2000: loss 1.367478
    iteration 300 / 2000: loss 1.328069
    iteration 400 / 2000: loss 1.302321
    iteration 500 / 2000: loss 1.291224
    iteration 600 / 2000: loss 1.333830
    iteration 700 / 2000: loss 1.268352
    iteration 800 / 2000: loss 1.328387
    iteration 900 / 2000: loss 1.216723
    iteration 1000 / 2000: loss 1.269007
    iteration 1100 / 2000: loss 1.317510
    iteration 1200 / 2000: loss 1.279888
    iteration 1300 / 2000: loss 1.312149
    iteration 1400 / 2000: loss 1.353501
    iteration 1500 / 2000: loss 1.364308
    iteration 1600 / 2000: loss 1.301765
    iteration 1700 / 2000: loss 1.303390
    iteration 1800 / 2000: loss 1.289426
    iteration 1900 / 2000: loss 1.366508
    iteration 0 / 2000: loss 1.429835
    iteration 100 / 2000: loss 1.445028
    iteration 200 / 2000: loss 1.552547
    iteration 300 / 2000: loss 1.456728
    iteration 400 / 2000: loss 1.393206
    iteration 500 / 2000: loss 1.438968
    iteration 600 / 2000: loss 1.306339
    iteration 700 / 2000: loss 1.466495
    iteration 800 / 2000: loss 1.494262
    iteration 900 / 2000: loss 1.499518
    iteration 1000 / 2000: loss 1.446250
    iteration 1100 / 2000: loss 1.520655
    iteration 1200 / 2000: loss 1.483932
    iteration 1300 / 2000: loss 1.356655
    iteration 1400 / 2000: loss 1.433732
    iteration 1500 / 2000: loss 1.434193
    iteration 1600 / 2000: loss 1.392030
    iteration 1700 / 2000: loss 1.397724
    iteration 1800 / 2000: loss 1.364778
    iteration 1900 / 2000: loss 1.449768
    iteration 0 / 2000: loss 1.228715
    iteration 100 / 2000: loss 1.151260
    iteration 200 / 2000: loss 1.143409
    iteration 300 / 2000: loss 1.085134
    iteration 400 / 2000: loss 1.097682
    iteration 500 / 2000: loss 1.163392
    iteration 600 / 2000: loss 1.159075
    iteration 700 / 2000: loss 1.090076
    iteration 800 / 2000: loss 1.071873
    iteration 900 / 2000: loss 1.147323
    iteration 1000 / 2000: loss 1.072428
    iteration 1100 / 2000: loss 1.016981
    iteration 1200 / 2000: loss 1.021893
    iteration 1300 / 2000: loss 1.119259
    iteration 1400 / 2000: loss 0.992877
    iteration 1500 / 2000: loss 1.053308
    iteration 1600 / 2000: loss 1.024200
    iteration 1700 / 2000: loss 0.911456
    iteration 1800 / 2000: loss 0.934868
    iteration 1900 / 2000: loss 1.068774
    iteration 0 / 2000: loss 1.442868
    iteration 100 / 2000: loss 1.514442
    iteration 200 / 2000: loss 1.460781
    iteration 300 / 2000: loss 1.311649
    iteration 400 / 2000: loss 1.507784
    iteration 500 / 2000: loss 1.306633
    iteration 600 / 2000: loss 1.404050
    iteration 700 / 2000: loss 1.455769
    iteration 800 / 2000: loss 1.264488
    iteration 900 / 2000: loss 1.343532
    iteration 1000 / 2000: loss 1.241713
    iteration 1100 / 2000: loss 1.378258
    iteration 1200 / 2000: loss 1.321738
    iteration 1300 / 2000: loss 1.259410
    iteration 1400 / 2000: loss 1.402515
    iteration 1500 / 2000: loss 1.358518
    iteration 1600 / 2000: loss 1.240150
    iteration 1700 / 2000: loss 1.277821
    iteration 1800 / 2000: loss 1.310006
    iteration 1900 / 2000: loss 1.329610
    iteration 0 / 2000: loss 1.359767
    iteration 100 / 2000: loss 1.480527
    iteration 200 / 2000: loss 1.609894
    iteration 300 / 2000: loss 1.488826
    iteration 400 / 2000: loss 1.420502
    iteration 500 / 2000: loss 1.345023
    iteration 600 / 2000: loss 1.461974
    iteration 700 / 2000: loss 1.361866
    iteration 800 / 2000: loss 1.494334
    iteration 900 / 2000: loss 1.604090
    iteration 1000 / 2000: loss 1.366507
    iteration 1100 / 2000: loss 1.324250
    iteration 1200 / 2000: loss 1.426203
    iteration 1300 / 2000: loss 1.500490
    iteration 1400 / 2000: loss 1.404023
    iteration 1500 / 2000: loss 1.336745
    iteration 1600 / 2000: loss 1.467405
    iteration 1700 / 2000: loss 1.508486
    iteration 1800 / 2000: loss 1.492049
    iteration 1900 / 2000: loss 1.301769
    learing_rate 2.500000e-01 reg 1.000000e-03 val_acc: 0.583000
    learing_rate 2.500000e-01 reg 3.000000e-03 val_acc: 0.599000
    learing_rate 2.500000e-01 reg 5.000000e-03 val_acc: 0.559000
    learing_rate 3.000000e-01 reg 1.000000e-03 val_acc: 0.601000
    learing_rate 3.000000e-01 reg 3.000000e-03 val_acc: 0.583000
    learing_rate 3.000000e-01 reg 5.000000e-03 val_acc: 0.562000
    learing_rate 3.500000e-01 reg 1.000000e-03 val_acc: 0.591000
    learing_rate 3.500000e-01 reg 3.000000e-03 val_acc: 0.581000
    learing_rate 3.500000e-01 reg 5.000000e-03 val_acc: 0.570000
    best validation accuracy achieved during cross-validation: 0.601000



```python
# Run your best neural net classifier on the test set. You should be able
# to get more than 55% accuracy.

test_acc = (best_net.predict(X_test_feats) == y_test).mean()
print(test_acc)
```

    0.553

---

Link to the complete code：[https://github.com/ctttt1119/cs231n](https://github.com/ctttt1119/cs231n)