title: "[cs231n]Optimization: Stochastic Gradient Descent"
date: 2015-04-19 00:52:19
categories: Open Course
tags: 
---
[lecture note](http://cs231n.github.io/optimization-1/) 
Not too much in this lecture if you know the basic knowledge about optimization. From my perspective, the most important thing is about(Gradient check). 
<!--more-->
In the Open Course UFLDL, it also introduces gradient check, you can learn from [here](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization)

Gradient check: numerically checking the derivatives computed by your code to make sure that your implementation is correct. Code in python
```python
def eval_numerical_gradient(f, x):
  """ 
  a naive implementation of numerical gradient of f at x 
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """ 

  fx = f(x) # evaluate function value at original point
  grad = np.zeros(x.shape)
  h = 0.00001

  # iterate over all indexes in x
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:

    # evaluate function at x+h
    ix = it.multi_index
    old_value = x[ix]
    x[ix] = old_value + h # increment by h
    fxh = f(x) # evalute f(x + h)
    x[ix] = old_value # restore to previous value (very important!)

    # compute the partial derivative
    grad[ix] = (fxh - fx) / h # the slope
    it.iternext() # step to next dimension

  return grad
  ```

  **Mini-batch gradient descent**
  **Stochastic Gradient Descent (SGD)**: usually for update the weight one sample a time. But now some people also use it to describe Mini-batch gradient descent