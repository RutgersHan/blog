title: "[cs231n]Neural Networks Part 3: Learning and Evaluation"
date: 2015-04-22 16:55:56
categories: Open Course
tags: 
---

##Gradient Check
Use the centered formula:
$$
\frac{df(x)}{dx} = \frac{f(x + h) - f(x)}{h} \hspace{0.1in} \text{(bad, do not use)}
$$
$$
\frac{df(x)}{dx} = \frac{f(x + h) - f(x - h)}{2h} \hspace{0.1in} \text{(use instead)}
<<<<<<< HEAD
$$

Use relative error for the comparison
=======
$$

