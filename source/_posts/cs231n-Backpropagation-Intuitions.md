title: "[cs231n]Backpropagation, Intuitions"
date: 2015-04-19 01:15:04
categories: Open Course
tags: 
---

[Lecture Note](http://cs231n.github.io/optimization-2/)
This lecture is very inspiring, actually it clears a haunting questions about the backpropogation.
I think this lecture is nice and concise. If you forget something, go back directly to the orinal lecture. Here just adding some images to remember. 
<!--more-->
###**Backpropagation using chain rule**: 
Basis: 
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 1.42.02 AM.png)
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 1.42.20 AM.png)
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 1.42.27 AM.png)
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 2.24.58 AM.png)
** MAX GATE** 
For the max function, the (sub)gradient is 1 on the input that was larger and 0 on the other input.
The max gate routes the gradient. Unlike the add gate which distributed the gradient unchanged to all its inputs, the max gate distributes the gradient (unchanged) to exactly one of its inputs (the input that had the highest value during the forward pass). This is because the local gradient for a max gate is 1.0 for the highest value, and 0.0 for all other values. In the example circuit above, the max operation routed the gradient of 2.00 to the z variable, which had a higher value than w, and the gradient on w remains zero

backpropogation 的时候就是求output对输入的梯度（终极问题：输出对输入的梯度），好好想一想chain rule，就能得到正好是根据chain rule来算back backpropogation得到的梯度值

backpropagation is a beautifully local process.
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 1.42.41 AM.png)
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 1.42.56 AM.png)



###**Backprop in practice**
One example: 
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-19 at 2.12.14 AM.png)
```python
x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator   #(1)
num = x + sigy # numerator                               #(2)
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator #(3)
xpy = x + y                                              #(4)
xpysqr = xpy**2                                          #(5)
den = sigx + xpysqr # denominator                        #(6)
invden = 1.0 / den                                       #(7)
f = num * invden # done! 
```
by the end of the expression we have computed the forward pass. Notice that we have structured the code in such way that it contains multiple intermediate variables, each of which are only simple expressions for which we already know the local gradients. Therefore, computing the backprop pass is easy: We'll go backwards and for every variable along the way in the forward pass (sigy, num, sigx, xpy, xpysqr, den, invden) we will have the same variable, but one that begins with a d, which will hold the gradient of that variable with respect to the output of the circuit. Additionally, note that every single piece in our backprop will involve computing the local gradient of that expression, and chaining it with the gradient on that expression with a multiplication.
```python
# backprop f = num * invden
dnum = invden # gradient on numerator                             #(8)
dinvden = num                                                     #(8)
# backprop invden = 1.0 / den 
dden = (-1.0 / (den**2)) * dinvden                                #(7)
# backprop den = sigx + xpysqr
dsigx = (1) * dden                                                #(6)
dxpysqr = (1) * dden                                              #(6)
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr                                        #(5)
# backprop xpy = x + y
dx = (1) * dxpy                                                   #(4)
dy = (1) * dxpy                                                   #(4)
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below  #(3)
# backprop num = x + sigy
dx += (1) * dnum                                                  #(2)
dsigy = (1) * dnum                                                #(2)
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy                                 #(1)
# done! phew
```

###Two important things:

**Cache forward pass variables**. To compute the backward pass it is very helpful to have some of the variables that were used in the forward pass. In practice you want to structure your code so that you cache these variables, and so that they are available during backpropagation. If this is too difficult, it is possible (but wasteful) to recompute them.

**Gradients add up at forks**. The forward expression involves the variables x,y multiple times, so when we perform backpropagation we must be careful to use += instead of = to accumulate the gradient on these variables (otherwise we would overwrite it). This follows the multivariable chain rule in Calculus, which states that if a variable branches out to different parts of the circuit, then the gradients that flow back to it will add.

**Unintuitive effects and their consequences**:  Go to the original lecture if you could not remember(not very important)
###Gradients for vectorized operations(different scaling problem)
**Matrix-Matrix multiply gradient**. Possibly the most tricky operation is the matrix-matrix multiplication (which generalizes all matrix-vector and vector-vector) multiply operations:
```python
# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape) # same shape as D
dW = dD.dot(X.T) #.T gives the transpose of the matrix
dX = W.T.dot(dD)
```
**Tip**: use dimension analysis! Note that you do not need to remember the expressions for dW and dX because they are easy to re-derive based on dimensions. For instance, we know that the gradient on the weights dW must be of the same size as W after it is computed, and that it must depend on matrix multiplication of X and dD (as is the case when both X,W are single numbers and not matrices). There is always exactly one way of achieving this so that the dimensions work out. For example, X is of size [10 x 3] and dD of size [5 x 3], so if we want dW and W has shape [5 x 10], then the only way of achieving this is with dD.dot(X.T), as shown above.
