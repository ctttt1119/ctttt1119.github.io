---
layout:     post
title:      cs231n
subtitle:   Dropout
date:       2020-03-13
author:     Shawn
header-img: img/Stanford.jpg
catalog: true
tags:
    - cs231n
















---

# Dropout

Dropout [1] is a technique for regularizing neural networks by randomly setting some output activations to zero during the forward pass. In this exercise you will implement a dropout layer and modify your fully-connected network to optionally use dropout.

[1] [Geoffrey E. Hinton et al, "Improving neural networks by preventing co-adaptation of feature detectors", arXiv 2012](https://arxiv.org/abs/1207.0580)


```python
# As usual, a bit of setup
from __future__ import print_function
import time
import numpy as np
import matplotlib.pyplot as plt
from cs231n.classifiers.fc_net import *
from cs231n.data_utils import get_CIFAR10_data
from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.solver import Solver

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

    The autoreload extension is already loaded. To reload it, use:
      %reload_ext autoreload



```python
# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.items():
  print('%s: ' % k, v.shape)
```

    X_train:  (49000, 3, 32, 32)
    y_train:  (49000,)
    X_val:  (1000, 3, 32, 32)
    y_val:  (1000,)
    X_test:  (1000, 3, 32, 32)
    y_test:  (1000,)


# Dropout forward pass
In the file `cs231n/layers.py`, implement the forward pass for dropout. Since dropout behaves differently during training and testing, make sure to implement the operation for both modes.

Once you have done so, run the cell below to test your implementation.


```python
np.random.seed(231)
x = np.random.randn(500, 500) + 10

for p in [0.25, 0.4, 0.7]:
  out, _ = dropout_forward(x, {'mode': 'train', 'p': p})
  out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})

  print('Running tests with p = ', p)
  print('Mean of input: ', x.mean())
  print('Mean of train-time output: ', out.mean())
  print('Mean of test-time output: ', out_test.mean())
  print('Fraction of train-time output set to zero: ', (out == 0).mean())
  print('Fraction of test-time output set to zero: ', (out_test == 0).mean())
  print()
```

    Running tests with p =  0.25
    Mean of input:  10.000207878477502
    Mean of train-time output:  10.014059116977283
    Mean of test-time output:  10.000207878477502
    Fraction of train-time output set to zero:  0.749784
    Fraction of test-time output set to zero:  0.0
    
    Running tests with p =  0.4
    Mean of input:  10.000207878477502
    Mean of train-time output:  9.977917658761159
    Mean of test-time output:  10.000207878477502
    Fraction of train-time output set to zero:  0.600796
    Fraction of test-time output set to zero:  0.0
    
    Running tests with p =  0.7
    Mean of input:  10.000207878477502
    Mean of train-time output:  9.987811912159426
    Mean of test-time output:  10.000207878477502
    Fraction of train-time output set to zero:  0.30074
    Fraction of test-time output set to zero:  0.0


​    

# Dropout backward pass
In the file `cs231n/layers.py`, implement the backward pass for dropout. After doing so, run the following cell to numerically gradient-check your implementation.


```python
np.random.seed(231)
x = np.random.randn(10, 10) + 10
dout = np.random.randn(*x.shape)

dropout_param = {'mode': 'train', 'p': 0.2, 'seed': 123}
out, cache = dropout_forward(x, dropout_param)
dx = dropout_backward(dout, cache)
dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)

# Error should be around e-10 or less
print('dx relative error: ', rel_error(dx, dx_num))
```

    dx relative error:  5.44560814873387e-11


## Inline Question 1:
What happens if we do not divide the values being passed through inverse dropout by `p` in the dropout layer? Why does that happen?

## Answer:
因为测试时，我们希望神经元的输出是与训练时一致的。以p=0.5为例，在测试时神经元必须把它们的输出减半，这是因为在训练的时候它们的输出只有一半。

会造成这种结果的原因：
先假设有一个神经元x的输出，那么进行随机失活的时候，该神经元的输出就是px+(1-p)x，这时有1-p的概率神经元的输出为0。

# Fully-connected nets with Dropout
In the file `cs231n/classifiers/fc_net.py`, modify your implementation to use dropout. Specifically, if the constructor of the network receives a value that is not 1 for the `dropout` parameter, then the net should add a dropout layer immediately after every ReLU nonlinearity. After doing so, run the following to numerically gradient-check your implementation.


```python
np.random.seed(231)
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for dropout in [1, 0.75, 0.5]:
  print('Running check with dropout = ', dropout)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            weight_scale=5e-2, dtype=np.float64,
                            dropout=dropout, seed=123)

  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)
  
  # Relative errors should be around e-6 or less; Note that it's fine
  # if for dropout=1 you have W2 error be on the order of e-5.
  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))
  print()
```

    Running check with dropout =  1
    Initial loss:  2.3004790897684924
    W1 relative error: 1.48e-07
    W2 relative error: 2.21e-05
    W3 relative error: 3.53e-07
    b1 relative error: 5.38e-09
    b2 relative error: 2.09e-09
    b3 relative error: 5.80e-11
    
    Running check with dropout =  0.75
    Initial loss:  2.302371489704412
    W1 relative error: 1.90e-07
    W2 relative error: 4.76e-06
    W3 relative error: 2.60e-08
    b1 relative error: 4.73e-09
    b2 relative error: 1.82e-09
    b3 relative error: 1.70e-10
    
    Running check with dropout =  0.5
    Initial loss:  2.3042759220785896
    W1 relative error: 3.11e-07
    W2 relative error: 1.84e-08
    W3 relative error: 5.35e-08
    b1 relative error: 2.58e-08
    b2 relative error: 2.99e-09
    b3 relative error: 1.13e-10


​    

# Regularization experiment
As an experiment, we will train a pair of two-layer networks on 500 training examples: one will use no dropout, and one will use a keep probability of 0.25. We will then visualize the training and validation accuracies of the two networks over time.


```python
# Train two identical nets, one with dropout and one without
np.random.seed(231)
num_train = 500
small_data = {
  'X_train': data['X_train'][:num_train],
  'y_train': data['y_train'][:num_train],
  'X_val': data['X_val'],
  'y_val': data['y_val'],
}

solvers = {}
dropout_choices = [1, 0.25]
for dropout in dropout_choices:
  model = FullyConnectedNet([500], dropout=dropout)
  print(dropout)

  solver = Solver(model, small_data,
                  num_epochs=25, batch_size=100,
                  update_rule='adam',
                  optim_config={
                    'learning_rate': 5e-4,
                  },
                  verbose=True, print_every=100)
  solver.train()
  solvers[dropout] = solver
  print()
```

    1
    (Iteration 1 / 125) loss: 7.856643
    (Epoch 0 / 25) train acc: 0.236000; val_acc: 0.190000
    (Epoch 1 / 25) train acc: 0.250000; val_acc: 0.178000
    (Epoch 2 / 25) train acc: 0.360000; val_acc: 0.217000
    (Epoch 3 / 25) train acc: 0.508000; val_acc: 0.242000
    (Epoch 4 / 25) train acc: 0.532000; val_acc: 0.241000
    (Epoch 5 / 25) train acc: 0.534000; val_acc: 0.272000
    (Epoch 6 / 25) train acc: 0.616000; val_acc: 0.265000
    (Epoch 7 / 25) train acc: 0.694000; val_acc: 0.246000
    (Epoch 8 / 25) train acc: 0.754000; val_acc: 0.282000
    (Epoch 9 / 25) train acc: 0.812000; val_acc: 0.294000
    (Epoch 10 / 25) train acc: 0.876000; val_acc: 0.311000
    (Epoch 11 / 25) train acc: 0.908000; val_acc: 0.283000
    (Epoch 12 / 25) train acc: 0.930000; val_acc: 0.279000
    (Epoch 13 / 25) train acc: 0.922000; val_acc: 0.291000
    (Epoch 14 / 25) train acc: 0.950000; val_acc: 0.293000
    (Epoch 15 / 25) train acc: 0.954000; val_acc: 0.310000
    (Epoch 16 / 25) train acc: 0.968000; val_acc: 0.294000
    (Epoch 17 / 25) train acc: 0.980000; val_acc: 0.305000
    (Epoch 18 / 25) train acc: 0.988000; val_acc: 0.307000
    (Epoch 19 / 25) train acc: 0.970000; val_acc: 0.304000
    (Epoch 20 / 25) train acc: 0.982000; val_acc: 0.288000
    (Iteration 101 / 125) loss: 0.104582
    (Epoch 21 / 25) train acc: 0.984000; val_acc: 0.307000
    (Epoch 22 / 25) train acc: 0.978000; val_acc: 0.319000
    (Epoch 23 / 25) train acc: 0.990000; val_acc: 0.330000
    (Epoch 24 / 25) train acc: 0.984000; val_acc: 0.316000
    (Epoch 25 / 25) train acc: 0.990000; val_acc: 0.298000
    
    0.25
    (Iteration 1 / 125) loss: 17.318480
    (Epoch 0 / 25) train acc: 0.216000; val_acc: 0.177000
    (Epoch 1 / 25) train acc: 0.300000; val_acc: 0.207000
    (Epoch 2 / 25) train acc: 0.396000; val_acc: 0.241000
    (Epoch 3 / 25) train acc: 0.422000; val_acc: 0.266000
    (Epoch 4 / 25) train acc: 0.498000; val_acc: 0.306000
    (Epoch 5 / 25) train acc: 0.532000; val_acc: 0.314000
    (Epoch 6 / 25) train acc: 0.550000; val_acc: 0.279000
    (Epoch 7 / 25) train acc: 0.596000; val_acc: 0.305000
    (Epoch 8 / 25) train acc: 0.560000; val_acc: 0.295000
    (Epoch 9 / 25) train acc: 0.634000; val_acc: 0.336000
    (Epoch 10 / 25) train acc: 0.646000; val_acc: 0.288000
    (Epoch 11 / 25) train acc: 0.642000; val_acc: 0.297000
    (Epoch 12 / 25) train acc: 0.654000; val_acc: 0.287000
    (Epoch 13 / 25) train acc: 0.712000; val_acc: 0.299000
    (Epoch 14 / 25) train acc: 0.728000; val_acc: 0.345000
    (Epoch 15 / 25) train acc: 0.750000; val_acc: 0.304000
    (Epoch 16 / 25) train acc: 0.750000; val_acc: 0.300000
    (Epoch 17 / 25) train acc: 0.772000; val_acc: 0.330000
    (Epoch 18 / 25) train acc: 0.786000; val_acc: 0.340000
    (Epoch 19 / 25) train acc: 0.824000; val_acc: 0.300000
    (Epoch 20 / 25) train acc: 0.828000; val_acc: 0.304000
    (Iteration 101 / 125) loss: 50.972035
    (Epoch 21 / 25) train acc: 0.850000; val_acc: 0.312000
    (Epoch 22 / 25) train acc: 0.846000; val_acc: 0.327000
    (Epoch 23 / 25) train acc: 0.820000; val_acc: 0.311000
    (Epoch 24 / 25) train acc: 0.844000; val_acc: 0.313000
    (Epoch 25 / 25) train acc: 0.876000; val_acc: 0.321000


​    


```python
# Plot train and validation accuracies of the two models

train_accs = []
val_accs = []
for dropout in dropout_choices:
  solver = solvers[dropout]
  train_accs.append(solver.train_acc_history[-1])
  val_accs.append(solver.val_acc_history[-1])

plt.subplot(3, 1, 1)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)
plt.title('Train accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')
  
plt.subplot(3, 1, 2)
for dropout in dropout_choices:
  plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)
plt.title('Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(ncol=2, loc='lower right')

plt.gcf().set_size_inches(15, 15)
plt.show()
```


![output_12_0.png](https://i.loli.net/2020/03/13/cQmuxwNMiA8doLP.png)


## Inline Question 2:
Compare the validation and training accuracies with and without dropout -- what do your results suggest about dropout as a regularizer?

## Answer:
使用了dropout后，训练集的准确率比未使用dropout的准确率低了一些，但是测试机的准确率更高，这说明了dropout可以作为regularization.


## Inline Question 3:
Suppose we are training a deep fully-connected network for image classification, with dropout after hidden layers (parameterized by keep probability p). If we are concerned about overfitting, how should we modify p (if at all) when we decide to decrease the size of the hidden layers (that is, the number of nodes in each layer)?

## Answer:
如果担心过拟合，而减小隐藏层大小（即每层的节点数），那么网络就变得更简单，我们假设要将p降低，即dropout的的部分增多，那么每次训练的神经元就很少很少了，最后结果就会欠拟合，所以需要将p增大。

----

Link to the complete code：[https://github.com/ctttt1119/cs231n](https://github.com/ctttt1119/cs231n)