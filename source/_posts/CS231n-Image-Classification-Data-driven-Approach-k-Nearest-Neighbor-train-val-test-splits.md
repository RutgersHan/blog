title: "[CS231n]Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits"
date: 2015-04-16 11:53:51
categories: Open Course
tags: 
---

[Lecture Website](http://cs231n.github.io/classification/)
Key words: L1/L2 distances, hyperparameter search, cross-validation, Nearest Neighbour 
<!--more-->

This tutorial is extremely good(class notes is enough. It contains all the materials in slides), better to read the original one. Here I just wrote some points, which I was not quite famaliar with, to remember. 

1. [CIFAR Dataset](http://www.cs.toronto.edu/~kriz/cifar.html): Entry-level dataset for image classification
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

2. [t-SNE](http://lvdmaaten.github.io/tsne/): popular visualization technique 
t-SNE for CIFAR-10 images using L2 distanceL 
![](http://7xikhz.com1.z0.glb.clouddn.com/pixels_embed_cifar10.jpg)

3. Validation sets for Hyperparameter tuning: The idea is to split our training set in two: a slightly smaller training set, and what we call a validation set. Using CIFAR-10 as an example, we could for example use 49,000 of the training images for training, and leave 1,000 aside for validation. By the end of this procedure, we could plot a graph that shows which values of k work best. We would then stick with this value and evaluate once on the actual test set.Here is what this might look like in the case of CIFAR-10:
```python
# assume we have Xtr_rows, Ytr, Xte_rows, Yte as before
# recall Xtr_rows is 50,000 x 3072 matrix
Xval_rows = Xtr_rows[:1000, :] # take first 1000 for validation
Yval = Ytr[:1000]
Xtr_rows = Xtr_rows[1000:, :] # keep last 49,000 for train
Ytr = Ytr[1000:]

# find hyperparameters that work best on the validation set
validation_accuracies = []
for k in [1, 3, 5, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr_rows, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval_rows, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
```
4. Cross Validation: In cases where the size of your training data (and therefore also the validation data) might be small, people sometimes use a more sophisticated technique for hyperparameter tuning called cross-validation. Working with our previous example, the idea is that instead of arbitrarily picking the first 1000 datapoints to be the validation set and rest training set, you can get a better and less noisy estimate of how well a certain value of k works by iterating over different validation sets and averaging the performance across these. For example, in 5-fold cross-validation, we would split the training data into 5 equal folds, use 4 of them for training, and 1 for validation. We would then iterate over which fold is the validation fold, evaluate the performance, and finally average the performance across the different folds.
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-16 at 12.24.20 PM.png)
**Exactly 5 points for each K **
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-16 at 12.24.10 PM.png)
**In Practice: if the training data size is small, go cross validation: **
 people prefer to avoid cross-validation in favor of having a single validation split, since cross-validation can be computationally expensive. The splits people tend to use is between 50%-90% of the training data for training and rest for validation. However, this depends on multiple factors: For example if the number of hyperparameters is large you may prefer to use bigger validation splits. If the number of examples in the validation set is small (perhaps only a few hundred or so), it is safer to use cross-validation. Typical number of folds you can see in practice would be 3-fold, 5-fold or 10-fold cross-validation.

 ** More about validation **
 *More parameters to tune, larger validation set or more folds*

 >Split your training data randomly into train/val splits. As a rule of thumb, between 70-90% of your data usually goes to the train split. This setting depends on how many hyperparameters you have and how much of an influence you expect them to have. If there are many hyperparameters to estimate, you should err on the side of having larger validation set to estimate them effectively. If you are concerned about the size of your validation data, it is best to split the training data into folds and perform cross-validation. If you can afford the computational budget it is always safer to go with cross-validation (the more folds the better, but more expensive).

 *Burn the validation set, not using it as the the training(e.g. 1NN don't use validation set as reference )*
 >Take note of the hyperparameters that gave the best results. There is a question of whether you should use the full training set with the best hyperparameters, since the optimal hyperparameters might change if you were to fold the validation data into your training set (since the size of the data would be larger). In practice it is cleaner to not use the validation data in the final classifier and consider it to be burned on estimating the hyperparameters. Evaluate the best model on the test set. Report the test set accuracy and declare the result to be the performance of the kNN classifier on your data.
