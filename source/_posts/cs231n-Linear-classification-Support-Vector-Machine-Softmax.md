title: "[cs231n]Linear classification: Support Vector Machine, Softmax"
date: 2015-04-16 12:46:32
categories: Open Course
tags: 

---

[lecture webpapge](http://cs231n.github.io/linear-classify/)
Key Words: parameteric approach, bias trick, hinge loss, cross-entropy loss, L2 regularization

<!--more-->

Again, the tutorial itself is very good and everyone should read it. It clears my confusions about some concepts. Here I just write the materials which seems inspiring to myself. 

###**Linear Classification**
**score function:** maps the raw data to class scores $f(x_i) = W x_i +b$
**loss function:** quantifies the agreement between the predicted scores and the ground truth labels(e.g. hinge loss used in SVM, $max(0, 1 - y_i f(x_i)))$
###**Linear classifier**
Here first introduce the terminology used in this blog:
$$f(x_i) = W x_i +b$$
where image $x_i$ a single column vector of shape [D x 1], The matrix W (of size [K x D]), and the vector b (of size [K x 1]) are the parameters of the function. K is equal to the number of classes.E.g.  In CIFAR-10(10 digit classification), $x_i$ contains all pixels in the i-th image flattened into a single [3072 x 1] column, $W$ is [10 x 3072] and b is [10 x 1]
####**Interpretation of linear classifiers as template matching**
interpretation for the weights W is that each row of W corresponds to a template (or sometimes also called a prototype) for one of the classes. The score of each class for an image is then obtained by comparing each template with the image using an inner product (or dot product) one by one to find the one that "fits" best.Another way to think of it is that we are still effectively doing Nearest Neighbor, but instead of having thousands of training images we are only using a single image per class (although we will learn it, and it does not necessarily have to be one of the images in the training set), and we use the (negative) inner product as the distance instead of the L1 or L2 distance.
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-16 at 7.07.44 PM.png)
####**Bias Trick**
First, we should have the bias term(*I forget this some time*), otherwise $x_i=0$ would always give score of zero regardless of the weights.
Instead of keep bias term explicitly, a trick is to let x=[x,1], W = [W,b]. (combine the two sets of parameters into a single matrix that holds both of them by extending the vector xi with one additional dimension that always holds the constant 1 - a default bias dimension)
$$f(x_i) = W x_i +b => f(x_i) = W x_i$$
With our CIFAR-10 example, $x_i$ is now [3073 x 1] instead of [3072 x 1] - (with the extra dimension holding the constant 1), and W is now [10 x 3073] instead of [10 x 3072]. The extra column that W now corresponds to the bias b. An illustration might help clarify:
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-16 at 7.23.04 PM.png)
####**Data Preprocessing(centering)**
we used the raw pixel values (which range from [0...255]) as features. In Machine Learning, it is a very common practice to always perform normalization (mean 0, std 1)of your input features (in the case of images, every pixel is thought of as a feature)

 1. Computing a mean image across the training images and subtracting it from every image(zero mean centering is arguably more important but we will have to wait for its justification until we understand the dynamics of gradient descent ?)
 2. common preprocessing is to scale each input feature so that its values range from [-1, 1]. (For images, it seems that we do not do this)

###**Loss Function**

**Multiclass Support Vector Machine Loss**(exend the traditional svm from 2 classes to multiclass): The SVM loss is set up so that the SVM "wants" the correct class for each image to a have a score higher than the incorrect classes by some fixed margin $\Delta$
Recall that for the i-th example we are given the pixels xi and the label yi that specifies the index of the correct class. The score function takes the pixels and computes the vector $f(x_i,W)$ of class scores. For example, the score for the j-th class is the j-th element: $f(x_i,W)_j$. The Multiclass SVM loss for the i-th example is then formalized as follows(the most popular one, others like one-verse-all,all-vs-all are not good compare to this one):

$$L\_i = \sum\_{j\neq y\_i} (0, f(x\_i,W)\_j - f(x\_i,W)\_{y\_i} + \Delta)$$
Example: 
Suppose that we have three classes that receive the scores $f(x_i,W)=[13,−7,11]$, and that the first class is the true class (i.e. $y_i=0$). Also assume that Δ (a hyperparameter we will go into more detail about soon) is 10. The expression above sums over all incorrect classes $(j \neq y_i)$, so we get two terms:
$$L_i = max(0, -7-13+10) + max(0,11-13+10)$$
The second term computes [11 - 13 + 10] which gives 8. That is, even though the correct class had a higher score than the incorrect class (13 > 11), it was not greater by the desired margin of 10. The difference was only 2, which is why the loss comes out to 8 (i.e. how much higher the difference would have to be to meet the margin). In summary, the SVM loss function wants the score of the correct class yi to be larger than the incorrect class scores by at least by $\Delta$(delta).
$max(0,-)$ is called **hinge loss**, $max(0,-)^2$ is called squared hinge loss 
The unsquared version is more standard, but in some datasets the squared hinge loss can work better. This can be determined during cross-validation.
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-17 at 12.10.10 AM.png)
**Regularization**： Better Generalization ability. 
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-17 at 12.17.26 AM.png)
Note that biases do not have the same effect since, unlike the weights, they do not control the strength of influence of an input dimension. Therefore, **it is common to only regularize the weights W but not the biases b**. However, in practice this often turns out to have a negligible effect. 
Code for Loss function(fully vectorized version)
```python
def L(X, y, W):
  """ 
  fully-vectorized implementation :
  - X holds all the training examples as columns (e.g. 3073 x 50,000 in CIFAR-10)
  - y is array of integers specifying correct class (e.g. 50,000-D array)
  - W are weights (e.g. 10 x 3073)
  """
  # evaluate loss over all examples in X without using any for loops
  # left as exercise to reader in the assignment
	delta = 1.0
	scores = W.dot(X)
	num = scores.shape[1]
	# remember interger indexing in python, very important(different from MATLAB)
	margins = np.maximum(0, scores - scores[y,np.arange(num)] + delta); 
	margins[y,np.arange(num)] = 0
	loss_i = np.sum(margins)
	return loss_i
```

####Practical Considerations
The hyperparameters $\Delta$ and $\lambda$ seem like two different hyperparameters, but in fact they both control the same tradeoff: The tradeoff between the data loss and the regularization loss in the objective. Therefore, we can  safely be set to $\Delta=1.0$ in all cases and just regularize W (changing $\lambda$). So we only have one hyperparameter in the above case. 

###**Logistic Regression**
It turns out that the SVM is one of two commonly seen classifiers. The other popular choice is the Softmax classifier, which has a different loss function. If you've heard of the binary Logistic Regression classifier before, the Softmax classifier is its generalization to multiple classes.  In the Softmax classifier, the function mapping $f(x_i;W) = W x_i$stays unchanged, but we now interpret these scores as the unnormalized log probabilities for each class and replace the hinge loss with a cross-entropy loss that has the form:
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-17 at 10.52.14 AM.png)
$f_j(z) = \frac{e^{z_j}}{\sum_k{e^{z_k}}}$ is called the softmax function: It takes a vector of arbitrary real-valued scores (in z) and squashes it to a vector of values between zero and one that sum to one
**Information theory view**: 
The cross-entropy between a "true" distribution p and an estimated distribution q is defined as(#Similiar as KL-divergence, cross-entropy can be viewed as the difference of two distributions. In this case, since the true distribution is fixed, cross-entropy is the same as KL divergence):
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 12.12.21 AM.png)
The Softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities,and the "true" distribution, which in this interpretation is the distribution where all probability mass is on the correct class (i.e. p=[0,…1,…,0] contains a single 1 at the yi -th position.). 
**Probabilistic interpretation**:
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 12.22.20 AM.png)
can be interpreted as the (normalized) probability assigned to the correct label yi given the image xi and parameterized by W. In the probabilistic interpretation, we are therefore minimizing the negative log likelihood of the correct class, which can be interpreted as performing Maximum Likelihood Estimation (MLE). The regularization term R(W) in the full loss function as coming from a Gaussian prior over the weight matrix W, where instead of MLE we are performing the Maximum a posteriori (MAP) estimation
**Practical issues: Numeric stability(#f 都减去一个最大值)**
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 12.25.59 AM.png)


When you're writing code for computing the Softmax function in pratice, the intermediate terms $e^{f\_{y\_i}}$ and 
$\sum_j{e^{f_j}}$may be very large due to the exponentials. Dividing large numbers can be numerically unstable, so it is important to use a normalization trick. Notice that if we multiply the top and bottom of the fraction by a constant C and push it into the sum, we get the following (mathematically equivalent) expression:

We are free to choose the value of C. This will not change any of the results, but we can use this value to improve the numerical stability of the computation. A common choice for C is to set $log C = -max_j f_j$. This simply states that we should shift the values inside the vector f so that the highest value is zero. In code:

```python
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
```
** Softmax VS SVM **
Softmax classifier provides "probabilities" for each class, but it is also fake. See original tutorial for details. In summary, both the scores come from SVM and softmax, he ordering of the scores is interpretable, but the absolute numbers (or their differences) technically are not.



