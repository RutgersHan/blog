title: "[cs231n]Neural Networks Part 2: Setting up the Data and the Loss"
date: 2015-04-21 22:17:26
categories: Open Course
tags: 
---
Data Preprocessing, Weight Initialization, Regularization (L2/L1/Maxnorm/Dropout), Loss functions
[Lecture Notes](http://cs231n.github.io/neural-networks-2/)
Some references:
Should read:
[Elastic net regularization](http://web.stanford.edu/~hastie/Papers/B67.2%20%282005%29%20301-320%20Zou%20&%20Hastie.pdf)
[Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
[Dropout Training as Adaptive Regularization](http://papers.nips.cc/paper/4882-dropout-training-as-adaptive-regularization.pdf)
[DropConnect](http://cs.nyu.edu/~wanli/dropc/)

others:
[Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852)
[Hierarchical Softmax](http://arxiv.org/pdf/1310.4546.pdf)

<!--more-->
##Data Preprocessing

*Take home message*: We mention PCA/Whitening in these notes for completeness, but these transformations are not used with Convolutional Networks. However, it is very important to zero-center the data, and it is common to see normalization of every pixel as well.
*Common pitfall*: An important point to make about the preprocessing is that any preprocessing statistics (e.g. the data mean) must only be computed on the training data, and then applied to the validation / test data. E.g. computing the mean and subtracting it from every image across the entire dataset and then splitting the data into train/val/test splits would be a mistake. Instead, the mean must be computed only over the training data and then subtracted equally from all splits (train/val/test).

First we assume the data matrix `X` is of size `[N x D]`(N is the number of data, D is their dimensionality).

**Mean subtraction**: subtracting the mean across every individual feature in the data, and has the geometric interpretation of centering the cloud of data around the origin along every dimension. For images, it just subtract the mean images of dataset. `X -= np.mean(X, axis = 0)`. With images specifically, for convenience it can be common to subtract a single value from all pixels `(e.g. X -= np.mean(X))`, or to do so separately across the three color channels.

**Normalization**: normalizing the data dimensions so that they are of approximately the same scale. Two common ways to do this

* Divide each dimension by its standard deviation, once it has been zero-centered: (X /= np.std(X, axis = 0)). Therefore, the variance for each dimension is just 1. 
* Another form of this preprocessing normalizes each dimension so that the min and max along the dimension is -1 and 1 respectively. It only makes sense to apply this preprocessing if you have a reason to believe that different input features have different scales (or units), but they should be of approximately equal importance to the learning algorithm. In case of images, the relative scales of pixels are already approximately equal (and in range from 0 to 255), so it is not strictly necessary to perform this additional preprocessing step.
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-21 at 10.48.21 PM.png)
**PCA and Whitening**: Another form of preprocessing.  In this process, the data is first centered as described above. Then, we can compute the covariance matrix that tells us about the correlation structure in the data
```python
# Assume input data matrix X of size [N x D]
X -= np.mean(X, axis = 0) # zero-center the data (important)
cov = np.dot(X.T, X) / X.shape[0] # get the data covariance matrix
```
We can get the eigenvectors of the covariance matrix either by SVD of X or eigen decomposition of covariance matrix. `U,S,V = np.linalg.svd(cov)`. Once we get the eigenvectors, the rotated data (PCA is just the rotation, since the eigenvectors are orthogonal and norm is 1)can be get by 
```python
Xrot = np.dot(X, U) # decorrelate the data
```
Sometimes, if the variance of some projected dimensions  are so small, we can omit these dimensions and do dimension reduction. A nice property of `np.linalg.svd` is that in its returned value U, the eigenvector columns are sorted by their eigenvalues.
```python
Xrot_reduced = np.dot(X, U[:,:100]) # Xrot_reduced becomes [N x 100]
```
After this operation, we would have reduced the original dataset of size [N x D] to one of size [N x 100], keeping the 100 dimensions of the data that contain the most variance.
After PCA, sometimes we will do whitening(e.g. in GB-RBM assumption, every variable is independent and follow the standard Guassian distribution) whitening operation takes the data in the eigenbasis and divides every dimension by the eigenvalue to normalize the scale. The whitened data will be a gaussian with zero mean and identity covariance matrix. This step would take the form:
```python
# whiten the data:
# divide by the eigenvalues (which are square roots of the singular values)
Xwhite = Xrot / np.sqrt(S + 1e-5)
```
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-21 at 10.48.05 PM.png)

## Weight Initialization
With proper data normalization it is reasonable to assume that approximately half of the weights will be positive and half of them will be negative. A reasonable-sounding idea then might be to set all the initial weights to zero. 
**Pitfall**: all zero initialization: Since in this way, the network is symmetric , w would all be the same. Bad!
**Small random numbers**
```python
W = 0.001* np.random.randn(D,H) #randn samples from a zero mean, unit standard deviation gaussian
```
**Calibrating the variances with 1/sqrt(n)**: One problem with the above suggestion is that the distribution of the outputs from a randomly initialized neuron has a variance that grows with the number of inputs. It turns out that we can normalize the variance of each neuron's output to 1 by scaling its weight vector by the square root of its fan-in (i.e. its number of inputs). That is, the recommended heuristic is to initialize each neuron's weight vector as: `w = np.random.randn(n) / sqrt(n)`, where n is the number of its inputs. This ensures that all neurons in the network initially have approximately the same output distribution and empirically improves the rate of convergence. (please read the original lecture note to get the intuition behind this)
**Calibrating the variances with sqrt(2.0)/ n **
It has been shown in paper [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](http://arxiv.org/abs/1502.01852).  Initialization specifically for **ReLU** neurons, reaching the conclusion that the variance of neurons in the network should be 2.0/n, is the current recommendation for use in practice.
```python
w = np.random.randn(n) * sqrt(2.0/n)
```
**Sparse initialization**: every neuron is randomly connected part of the neurons below it. 
**Initializing the biases**: it is more common to simply use 0 bias initialization

##Regularization
**L2 Regularization**: has the intuitive interpretation of heavily penalizing peaky weight vectors and preferring diffuse weight vectors(try to use every feature). `W += -lambda * W` is also viewed as weight decay. 

** L1 or Elastic net Regularization**: make W sparse (for feature selection).  In practice, if you are not concerned with explicit feature selection, L2 regularization can be expected to give superior performance over L1. 

** Max norm constraints**:  clamping the weight vector $|x|_2 < c$  of every neuron to satisfy. Typical values of c are on orders of 3 or 4. 
** Dropout**: While training, dropout is implemented by only keeping a neuron active with some probability p (a hyperparameter), or setting it to zero otherwise.
* During training, Dropout can be interpreted as sampling a Neural Network within the full Neural Network, and only updating the parameters of the sampled network based on the input data. 
* During testing there is no dropout applied, with the interpretation of evaluating an averaged prediction across the exponentially-sized ensemble of all sub-networks (more about ensembles in the next section).
```python
""" Vanilla Dropout: Not recommended implementation (see notes below) """

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  """ X contains the data """

  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = np.random.rand(*H1.shape) < p # first dropout mask
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = np.random.rand(*H2.shape) < p # second dropout mask
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3

  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)

def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) * p # NOTE: scale the activations
  H2 = np.maximum(0, np.dot(W2, H1) + b2) * p # NOTE: scale the activations
  out = np.dot(W3, H2) + b3
 ```
The undesirable property of the scheme presented above is that we must scale the activations by p at test time. Since test-time performance is so critical, it is always preferable to use inverted dropout, which performs the scaling at train time, leaving the forward pass at test time untouched. Additionally, this has the appealing property that the prediction code can remain untouched when you decide to tweak where you apply dropout, or if at all. Inverted dropout looks as follows:
```python
""" 
Inverted Dropout: Recommended implementation example.
We drop and scale at train time and don't do anything at test time.
"""

p = 0.5 # probability of keeping a unit active. higher = less dropout

def train_step(X):
  # forward pass for example 3-layer neural network
  H1 = np.maximum(0, np.dot(W1, X) + b1)
  U1 = (np.random.rand(*H1.shape) < p) / p # first dropout mask. Notice /p!
  H1 *= U1 # drop!
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  U2 = (np.random.rand(*H2.shape) < p) / p # second dropout mask. Notice /p!
  H2 *= U2 # drop!
  out = np.dot(W3, H2) + b3

  # backward pass: compute gradients... (not shown)
  # perform parameter update... (not shown)

def predict(X):
  # ensembled forward pass
  H1 = np.maximum(0, np.dot(W1, X) + b1) # no scaling necessary
  H2 = np.maximum(0, np.dot(W2, H1) + b2)
  out = np.dot(W3, H2) + b3
```
**Theme of noise in forward pass**: Dropout falls into a more general category of methods that introduce stochastic behavior in the forward pass of the network. During testing, the noise is marginalized over analytically (as is the case with dropout when multiplying by p), or numerically (e.g. via sampling, by performing several forward passes with different random decisions and then averaging over them). One example: [DropConnect](http://cs.nyu.edu/~wanli/dropc/)
**Bias regularization**: Usually we don't regularize bias, However, in practical applications (and with proper data preprocessing) regularizing the bias rarely leads to significantly worse performance.
**In practice **: It is most common to use a single, global L2 regularization strength that is cross-validated. It is also common to combine this with dropout applied after all layers. The value of p=0.5 is a reasonable default, but this can be tuned on validation data.

##Loss functions
**Classification**:
- Multi-SVM loss(we can also square it, squared hinge loss): 
$$L\_i = \sum\_{j\neq y\_i} (0, f(x\_i,W)\_j - f(x\_i,W)\_{y\_i} + 1)$$
- Softmax loss: 
$$L_i = -log(\frac{e^{z_j}}{\sum_k{e^{z_k}}})$$
- **Problem: Large number of classes**. When the set of labels is very large (e.g. words in English dictionary, or ImageNet which contains 22,000 categories), it may be helpful to use [Hierarchical Softmax](http://arxiv.org/pdf/1310.4546.pdf) . The hierarchical softmax decomposes labels into a tree. Each label is then represented as a path along the tree, and a Softmax classifier is trained at every node of the tree to disambiguate between the left and right branch. The structure of the tree strongly impacts the performance and is generally problem-dependent.

**Attribute classification**: the above losses assume that there is a single correct answer y_i. But for attribute prediction, for each image, it might have many attributes. So we should deal with each attribute separately(e.g. binary classfication for each tag)

A sensible approach in this case is to build a binary classifier for every single attribute independently. For example, a binary classifier for each category independently would take the form:

$$
L\_i = \sum\_j \max(0, 1 - y\_{ij} f\_j)
$$


An alternative to this loss would be to train a logistic regression classifier for every attribute independently. A binary logistic regression classifier has only two classes (0,1), and calculates the probability of class 1 as:

$$
P(y = 1 \mid x; w, b) = \frac{1}{1 + e^{-(w^Tx +b)}} = \sigma (w^Tx + b)
$$

$$
L\_i = \sum\_j y\_{ij} \log(\sigma(f\_j)) + (1 - y\_{ij}) \log(1 - \sigma(f\_j))
$$

where the labels \\(y\_{ij}\\) are assumed to be either 1 (positive) or 0 (negative), and \\(\sigma(\cdot)\\) is the sigmoid function. The expression above can look scary but the gradient on \\(f\\) is in fact extremely simple and intuitive: \\(\partial{L\_i} / \partial{f\_j} = y\_{ij} - \sigma(f\_j)\\) (as you can double check yourself by taking the derivatives).

**Regression**:  the task of predicting real-valued quantities, use L2 norm( or maybe L1 norm)
$$
L\_i = \Vert f - y\_i \Vert\_2^2
$$

**Word of caution**: It is important to note that the L2 loss is much harder to optimize than a more stable loss such as Softmax. Intuitively, it requires a very fragile and specific property from the network to output exactly one correct value for each input (and its augmentations). Notice that this is not the case with Softmax, where the precise value of each score is less important: It only matters that their magnitudes are appropriate. Additionally, the L2 loss is less robust because outliers can introduce huge gradients. When faced with a regression problem, first consider if it is absolutely inadequate to quantize the output into bins. For example, if you are predicting star rating for a product, it might work much better to use 5 independent classifiers for ratings of 1-5 stars instead of a regression loss. Classification has the additional benefit that it can give you a distribution over the regression outputs, not just a single output with no indication of its confidence. If you're certain that classification is not appropriate, use the L2 but be careful: For example, the L2 is more fragile and applying dropout in the network (especially in the layer right before the L2 loss) is not a great idea.
**Structured prediction**. The structured loss refers to a case where the labels can be arbitrary structures such as graphs, trees, or other complex objects

