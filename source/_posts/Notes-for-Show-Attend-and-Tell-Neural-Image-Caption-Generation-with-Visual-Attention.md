title: "Notes for <Show, Attend and Tell: Neural Image Caption Generation with Visual Attention>"
date: 2015-04-14 15:57:10
categories: Research
tags: [Image Caption Generation, Deep Learning] 
---

Short Summary:
Use CNN feature and LSTM to learn to fix attention to a particular part of image while generating the corresponding words. Need to revisit this paper after a better understand of RNN for image caption analysis. 
![Overview](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-14 at 4.50.19 PM.png)
![Interesting Results](http://7xikhz.com1.z0.glb.clouddn.com/Screen Shot 2015-04-14 at 4.50.53 PM.png)
<!--more-->

####Model Details
* Encoder: lower convolutional layer(the fourth layer),since it only focus on certain parts of the image
* Decoder: LSTM(context vector, the previous hidden state and the previously generated words.) [LSTM](http://arxiv.org/pdf/1409.2329v5.pdf)

####Stochastic "Hard" vs Deterministic "Soft"
The way to get context vector is important, which separates these two methods.
* For the Stochastic Hard one, we assume for each word, the context is only a particular attention region. The attention location is a intermediate latent variable. Then sampling method is used for the inference in the optimization problem. 
* For Deterministic "Soft" one, the context is soft combination of all the features in different locations. In that way, we can learn the contribution of each location in the specific given word. 

####Training Procedure
1. Use Oxford net pre-trained on ImageNet without fine tuning.  Using the fourth layer
2. Regularization: Dropout and early stopping on BLEU score.   

#### Visualization
The original image is $ 224 \* 224 $, the output convolution layer is $14 \* 14$(224/14=16), so we upsample the weights(the soft combination weights) by a factor of 16 and apply Gaussian fileter as the output image. 

