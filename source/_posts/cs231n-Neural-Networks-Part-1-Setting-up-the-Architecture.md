title: "[cs231n]Neural Networks Part 1: Setting up the Architecture "
date: 2015-04-21 16:51:11
categories: Open Course
tags: 
---
[Lecture Notes](http://cs231n.github.io/neural-networks-1/)
Useful Refenrences
* [Deep Learning book in press by Bengio](http://www.iro.umontreal.ca/~bengioy/dlbook/)
* [Do Deep Nets Really Need to be Deep?](http://arxiv.org/abs/1312.6184)
* [FitNets: Hints for Thin Deep Nets](http://arxiv.org/abs/1412.6550)
* [The Loss Surfaces of Multilayer Networks](http://arxiv.org/abs/1412.0233)
* [deeplearning.net tutorial with Theano](http://www.deeplearning.net/tutorial/mlp.html)
* [ConvNetJS demos for intuitions](http://cs.stanford.edu/people/karpathy/convnetjs/)
* [Michael Nielsen's](http://neuralnetworksanddeeplearning.com/chap1.html)

<!--more-->
## Biological neuron VS compuational neuron

* Dendrites in biological neurons perform complex nonlinear computations, whereas computational neurons are linear functions of input($f = Wx$)
* Snapses are not just a single weight, they're a complex non-linear dynamical system, whereas computational neurons the Snapses is just modeled a parameter($w_i$)
* The exact timing of the output spikes in neurons systems is known to be important, whereas computional neurons are give the activition rate(activation function)

## Different activition functions
**Take Home Message about this section** : 
 - "What neuron type should I use?" Use the ReLU non-linearity, be careful with your learning rates and possibly monitor the fraction of "dead" units in a network. If this concerns you, give Leaky ReLU or Maxout a try. Never use sigmoid. Try tanh, but expect it to work worse than ReLU/Maxout.
 - It is very rare to mix and match different types of neurons in the same network, even though there is no fundamental problem with doing so.

1. **Sigmoid**(was very popular, but not now, two major drawbacks): 
	- Sigmoids saturate and kill gradients(in the area that far away from zero)
	- Sigmoid outputs are not zero-centered. This could introduce undesirable zig-zagging dynamics in the gradient updates for the weights.
2. **Tanh** 
	- Tanh saturate and kill gradients(in the area that far away from zero)
	- The outputs is zero-centered
	- In practice the tanh non-linearity is always preferred to the sigmoid nonlinearity.
3. **ReLU** ($f(x) = max(0,x)$)
	- (+) Greatly accelerate (factor of 6 in Alex net) the convergence of stochastic gradient descent compared to the sigmoid/tanh functions. It is argued that this is due to its linear, non-saturating form.
	- (+) Simple calculation. Compared to tanh/sigmoid neurons that involve expensive operations (exponentials, etc.), the ReLU can be implemented by simply thresholding a matrix of activations at zero.
	- (-) ReLU can "die" during training. E.g. a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on. That is, the ReLU units can irreversibly die during training since they can get knocked off the data manifold. For example, you may find that as much as 40% of your network can be "dead" if the learning rate is set too high. With a proper setting of the learning rate this is less frequently an issue.
4. **Leaky ReLU** Leaky ReLUs are one attempt to fix the "dying ReLU" problem. Instead of the function being zero when x < 0, a leaky ReLU will instead have a small negative slope (of 0.01, or so). That is, the function computes f(x)=𝟙(x<0)(αx)+𝟙(x>=0)(x) where α is a small constant. Some people report success with this form of activation function, but the results are not always consistent.

5. **Maxout** $f(x) = max(w^T_1x+b_1,w^T_2x+b_2)$: 
	- Do not have the functional form $f(w^Tx+b)$ where a non-linearity is applied on the dot product between the weights and the data. 
	- Both ReLU and Leaky ReLU are a special case of Maxout(for example, for ReLU we have w1,b1=0).The Maxout neuron therefore enjoys all the benefits of a ReLU unit (linear regime of operation, no saturation) and does not have its drawbacks (dying ReLU)
	- It doubles the number of parameters for every single neuron, leading to a high total number of parameters.

## Neural Network architectures
![](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-21 at 5.32.49 PM.png)
###Naming conventions
Notice that when we say N-layer neural network, we do not count the input layer(But we count the output layer). Just as above image, the left is 2-layer nerual network and the right is a three layer neural network. You may also hear these "Artificial Neural Networks" (ANN) or "Multi-Layer Perceptrons" (MLP). They are just the same as "Neural Network"
### Output layer
Output layer don't have the activation function. This is because the last output layer is usually taken to represent the class scores (e.g. in classification), which are arbitrary real-valued numbers, or some kind of real-valued target (e.g. in regression)
### Sizing neural networks. 
The two metrics that people commonly use to measure the size of neural networks are the number of neurons, or more commonly the number of parameters. For the above example: 
* The first network (left) has 4 + 2 = 6 neurons (not counting the inputs), [3 x 4] + [4 x 2] = 20 weights and 4 + 2 = 6 biases, for a total of 26 learnable parameters.
* The second network (right) has 4 + 4 + 1 = 9 neurons, [3 x 4] + [4 x 4] + [4 x 1] = 12 + 16 + 4 = 32 weights and 4 + 4 + 1 = 9 biases, for a total of 41 learnable parameters.
* Modern Convolutional Networks contain on orders of 100 million parameters and are usually made up of approximately 10-20 layers (hence deep learning). However, as we will see the number of effective connections is significantly greater due to parameter sharing.

### Example feed-forward computation
```python
# forward-pass of a 3-layer neural network:
f = lambda x: 1.0/(1.0 + np.exp(-x)) # activation function (use sigmoid)
x = np.random.randn(3, 1) # random input vector of three numbers (3x1)
h1 = f(np.dot(W1, x) + b1) # calculate first hidden layer activations (4x1)
h2 = f(np.dot(W2, h1) + b2) # calculate second hidden layer activations (4x1)
out = np.dot(W3, h2) + b3 # output neuron (1x1)
```
### Representational power

What is the representational power of this family of functions? In particular, are there functions that cannot be modeled with a Neural Network?

* Neural Networks with at least one hidden layer are universal approximators
* If one hidden layer suffices to approximate any function, why use more layers and go deeper? The answer is that the fact that a two-layer Neural Network is a universal approximator is, while mathematically cute, a relatively weak and useless statement in practice.
* In practice it is often the case that 3-layer neural networks will outperform 2-layer nets, but going even deeper (4,5,6-layer) rarely helps much more. In contrast, for Convolutional Networks, where depth has been found to be an extremely important component for a good recognition system(on order of 10 learnable layers. One argument for this observation is that images contain hierarchical structure (e.g. faces are made up of eyes, which are made up of edges, etc.), so several layers of processing make intuitive sense for this data domain.

### Setting number of layers and their sizes
* Neural Networks with more neurons can express more complicated functions. But more neurons(more parameters) can cause overfitting problems(learn the outliers/noise of the training dataset) if without enough training data. 

* **It seems that smaller neural networks can be preferred if the data is not complex enough to prevent overfitting. However, this is incorrect - there are many other preferred ways to prevent overfitting in Neural Networks that we will discuss later (such as L2 regularization, dropout, input noise). In practice, it is always better to use these methods to control overfitting instead of the number of neurons.**
* The subtle reason behind this is that smaller networks are harder to train with local methods such as Gradient Descent: It's clear that their loss functions have relatively few local minima, but it turns out that many of these minima are easier to converge to, and that they are bad (i.e. with high loss). Conversely, bigger neural networks contain significantly more local minima, but these minima turn out to be much better in terms of their actual loss. 

