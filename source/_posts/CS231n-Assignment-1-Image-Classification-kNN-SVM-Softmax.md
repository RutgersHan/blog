title: "[CS231n]Assignment #1: Image Classification, kNN, SVM, Softmax"
date: 2015-04-19 15:50:34
categories: Open Course
tags: 
---
KNN, SVM, Softmax
<!--more-->
## KNN
###Vectorized version to compute distance for KNN
```python
def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    pass
    cross_product = X.dot(self.X_train.T);
    test_squared = np.sum(X**2, axis=1)
    train_squared = np.sum(self.X_train**2, axis=1)
    temp = -2 * cross_product + train_squared.reshape((1, num_train))
    temp = temp + test_squared.reshape((num_test,1));
    dists = np.sqrt(temp)

    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists
```
```python
# Let's compare how fast the implementations are
def time_function(f, *args):
  """
  Call a function f with args and return the time (in seconds) that it took to execute.
  """
  import time
  tic = time.time()
  f(*args)
  toc = time.time()
  return toc - tic

two_loop_time = time_function(classifier.compute_distances_two_loops, X_test)
print 'Two loop version took %f seconds' % two_loop_time

one_loop_time = time_function(classifier.compute_distances_one_loop, X_test)
print 'One loop version took %f seconds' % one_loop_time

no_loop_time = time_function(classifier.compute_distances_no_loops, X_test)
print 'No loop version took %f seconds' % no_loop_time

# you should see significantly faster performance with the fully vectorized implementation
```
> Two loop version took 70.987422 seconds
> One loop version took 50.101490 seconds
> No loop version took 0.824264 seconds

###Cross Validation
```python
num_folds = 5
k_choices = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

X_train_folds = []
y_train_folds = []
################################################################################
# TODO:                                                                        #
# Split up the training data into folds. After splitting, X_train_folds and    #
# y_train_folds should each be lists of length num_folds, where                #
# y_train_folds[i] is the label vector for the points in X_train_folds[i].     #
# Hint: Look up the numpy array_split function.                                #
################################################################################
pass
nums_train = X_train.shape[0]
orders = np.random.permutation(nums_train)
X_train_folds = np.array_split(X_train[orders],num_folds,axis=0)

y_train_orders  = y_train[orders];
y_train_orders = y_train_orders.reshape((nums_train,1))

y_train_folds = np.array_split(y_train_orders,num_folds,axis=0)

################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# A dictionary holding the accuracies for different values of k that we find
# when running cross-validation. After running cross-validation,
# k_to_accuracies[k] should be a list of length num_folds giving the different
# accuracy values that we found when using that value of k.
k_to_accuracies = {}


################################################################################
# TODO:                                                                        #
# Perform k-fold cross validation to find the best value of k. For each        #
# possible value of k, run the k-nearest-neighbor algorithm num_folds times,   #
# where in each case you use all but one of the folds as training data and the #
# last fold as a validation set. Store the accuracies for all fold and all     #
# values of k in the k_to_accuracies dictionary.                               #
################################################################################
pass

for i in range(num_folds):
    t_indexs = range(num_folds);
    t_indexs.remove(i)
    c_X_train = np.vstack([X_train_folds[j] for j in t_indexs])
    c_y_train = np.vstack([y_train_folds[j] for j in t_indexs])
    
    c_X_test = X_train_folds[i]
    c_y_test = y_train_folds[i]
    num_test = c_y_test.shape[0]
    for k in k_choices:
        if k not in k_to_accuracies:
            k_to_accuracies[k] = []
        classifier = KNearestNeighbor()
        classifier.train(c_X_train, c_y_train)
        dists = classifier.compute_distances_no_loops(c_X_test)
        y_test_pred = classifier.predict_labels(dists, k)
        y_test_pred = y_test_pred.reshape((num_test,1))
        num_correct = np.sum(y_test_pred == c_y_test)
        accuracy = float(num_correct) / num_test
        k_to_accuracies[k].append(accuracy)
        
################################################################################
#                                 END OF YOUR CODE                             #
################################################################################

# Print out the computed accuracies
for k in sorted(k_to_accuracies):
    for accuracy in k_to_accuracies[k]:
        print 'k = %d, accuracy = %f' % (k, accuracy)
```

###Plot the result for cross validation
```python
# plot the raw observations
for k in k_choices:
  accuracies = k_to_accuracies[k]
  plt.scatter([k] * len(accuracies), accuracies)

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(k_to_accuracies.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(k_to_accuracies.items())])
plt.errorbar(k_choices, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
```
## SVM
### **Vectorized version for multiclass SVM to compute loss and gradient**
```python
def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  scores = W.dot(X)
  num_classes = W.shape[0]
  num_train = X.shape[1]

  correct_class_score = scores[y, np.arange(num_train)]
  margin = scores - correct_class_score + 1
  margin[margin < 0] = 0
  margin[y, np.arange(num_train)] = 0
  loss = np.sum(margin) / num_train
  loss += 0.5 * reg * np.sum(W * W)


  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  margin_flag = margin>0;
  term2 = margin_flag.dot(X.T)

  margin_flag_sum = np.sum(margin_flag,axis=0)
  coeff_matrix = np.zeros((num_classes,num_train))
  coeff_matrix[y, np.arange(num_train)] = margin_flag_sum
  term1 = -coeff_matrix.dot(X.T)
  dW = term1 + term2
  dW = dW / num_train
  dW += reg * W



  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
```
### How to get training batch
```python
   for it in xrange(num_iters):
      X_batch = None
      y_batch = None
      indexs = np.random.choice(num_train,batch_size,False)
      y_batch = y[indexs]
      X_batch = X[:,indexs]
      # evaluate loss and gradient
      loss, grad = self.loss(X_batch, y_batch, reg)
      loss_history.append(loss)

```
### How to tune Hyperparameter
```python
learning_rates = [1e-7, 1e-6,1e-5]
regularization_strengths = [1e3, 5e4, 1e5]

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1   # The highest validation accuracy that we have seen so far.
best_svm = None # The LinearSVM object that achieved the highest validation rate.


for learning_rate in learning_rates:
    for reg in regularization_strengths:
        v_svm = LinearSVM()
        loss_hist = v_svm.train(X_train, y_train, learning_rate, reg,
                      num_iters=1500, verbose=True)
        y_train_pred = v_svm.predict(X_train)
        y_val_pred = v_svm.predict(X_val)
        y_train_accuracy = np.mean(y_train == y_train_pred)
        y_val_accuracy = np.mean(y_val == y_val_pred)
        results[(learning_rate,reg)] = (y_train_accuracy, y_val_accuracy)
        if y_val_accuracy>best_val:
            best_val = y_val_accuracy
            best_svm = v_svm
```
### Results of SVM
linear SVM on raw pixels final test set accuracy: 0.375000
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-21 at 1.18.24 PM.png)

## SoftMAX

For the loss of SoftMax  it is easy. Just add up the loss of all samples. 
To compute the gradient of W in Softmax, we need to do some math. Similiar as SVM, for dW it contains two parts

part1: 
$$
\nabla\_{w\_{y\_i}} L\_i = (\frac{e^{f\_{y\_i}}}{ \sum\_j e^{f\_j} } - 1) \times x_i
$$
part2: 

$$
\nabla\_{w\_{h \neq y\_i}} L\_i = (\frac{e^{f\_{h}}}{ \sum\_j e^{f\_j} }) \times x_i
$$

where $f_h = w_h x_i$


The vectorized verion to compute the loss and gradient. 
```python
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_classes = W.shape[0]
  num_train = X.shape[1]
  coeff1 = np.zeros((num_classes,num_train))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
 

  pass
  scores = W.dot(X)
  #numeric stability
  scores = scores - scores.max(0)
  exp_scores = np.exp(scores)
  sum_exp_scores = exp_scores.sum(0)
  normalized_scores = exp_scores / (sum_exp_scores)
  normalized_correct_scores = normalized_scores[y,np.arange(num_train)]
  log_cost = -np.log(normalized_correct_scores + 1e-5)
  loss =  np.sum(log_cost) / num_train + 0.5 * reg * np.sum(W * W) 
  

  coeff2 = normalized_scores.copy()
  coeff2[y,np.arange(num_train)] = 0


  dW_part2 = coeff2.dot(X.T)
  coeff1[y,np.arange(num_train)] = normalized_correct_scores - 1
  dW_part1 = coeff1.dot(X.T)
  dW =  (dW_part1 + dW_part2) / num_train + reg * W
  


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
```
