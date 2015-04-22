title: "Some common functions in Numpy and Scipy "
date: 2015-04-19 13:22:50
categories: Engineering
tags: 
---
import numpy as np
import scipy as scp

* **np.flatnonzero**  # return indices that are non-zero in the flattened version of a.
This is equivalent to a.ravel().nonzero()[0].
> see also
> nonzero
> Return the indices of the non-zero elements of the input array.
> ravel
> Return a 1-D array containing the elements of the input array.

* **numpy.random.choice(a, size=None, replace=True, p=None)  #Generates a random sample from a given 1-D array
>a   1-D array-like or int
> If an ndarray, a random sample is generated from its elements. If an int, the random sample is generated as if a was np.arange(n)

* ** numpy.argsort(a, axis=-1, kind='quicksort', order=None)[source]
Returns the indices that would sort an array.
* **np.random.permutation**
* **np.array_split**
* **np.vstack**