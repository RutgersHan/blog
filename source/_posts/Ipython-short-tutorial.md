title: "Ipython Notebook short tutorial"
date: 2015-04-19 12:57:23
categories: Engineering
tags: 
---
* Start Ipython notebook 
Ipython notebook.

* Use markdown cell, code cell, raw text cell for editing

* short cuts: 
Shift-Enter: run cell
Ctrl-Enter: run cell in-place
Alt-Enter: run cell, insert below
<!--more-->

*Plotting
The document says it needs to run %matplotlib before running matplotlib, but it seems now that we don't need that. 
```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```