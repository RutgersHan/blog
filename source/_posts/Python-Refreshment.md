title: "Python Refreshment"
date: 2015-04-15 12:35:02
categories: Engineering
tags: 
---
Have not using Python for months. It is annoying to check the details every time when start picking up python. Therefore, I decide to write down some remiders (Most are just copied from the following tutorials). 

<!--more-->

* [Python tutorial](http://cs231n.github.io/python-numpy-tutorial/)
* [Numpy for MATLAB User](http://wiki.scipy.org/NumPy_for_Matlab_Users)

###Basic Python
1. Python implements all of the usual operators for Boolean logic, but uses English words rather than symbols (&&, ||, etc.):
```python
t = True
f = False
print type(t) # Prints "<type 'bool'>"
print t and f # Logical AND; prints "False"
print t or f  # Logical OR; prints "True"
print not t   # Logical NOT; prints "False"
print t != f  # Logical XOR; prints "True" 

```
2. String
```python
hello = 'hello'   # String literals can use single quotes
world = "world"   # or double quotes; it does not matter.
print hello       # Prints "hello"
print len(hello)  # String length; prints "5"
hw = hello + ' ' + world  # String concatenation
print hw  # prints "hello world"
hw12 = '%s %s %d' % (hello, world, 12)  # sprintf style string formatting
print hw12  # prints "hello world 12"
s = "hello"
print s.capitalize()  # Capitalize a string; prints "Hello"
print s.upper()       # Convert a string to uppercase; prints "HELLO"
print s.rjust(7)      # Right-justify a string, padding with spaces; prints "  hello"
print s.center(7)     # Center a string, padding with spaces; prints " hello "
print s.replace('l', '(ell)')  # Replace all instances of one substring with another;
                               # prints "he(ell)(ell)o"
print '  world '.strip()  # Strip leading and trailing whitespace; prints "world"
```
3. List: A list is the Python equivalent of an array, but is resizeable and can contain elements of different types. Index start from 0:
```python 
xs = [3, 1, 2]   # Create a list
print xs, xs[2]  # Prints "[3, 1, 2] 2"
print xs[-1]     # Negative indices count from the end of the list; prints "2"
xs[2] = 'foo'    # Lists can contain elements of different types
print xs         # Prints "[3, 1, 'foo']"
xs.append('bar') # Add a new element to the end of the list
print xs         # Prints 
x = xs.pop()     # Remove and return the last element of the list
print x, xs      # Prints "bar [3, 1, 'foo']"
a = [1,2,3]
a.extend([9,8,7]) # a = [1,2,3,9,8,7]
a = [1,2,3]
a.append([9,8,7])  $ a = [1, 2, 3, [9, 8, 7]]
a.index(8)     #index of first occurrence
a.count(8)     #number of occurrences
a.reverse()    # reverse
a.sort()       # sort
a.sort(some_function)
```
3. For mutable objects(lists,np.array, dictionaries, user-fined objects,sets), "=" is just the reference 
```python
a = [1,2,3]
b = a
a.append(4)
print a  #[1, 2, 3, 4]
print b  #[1, 2, 3, 4]
```
4. Indexing / Sclicing: We will see slicing again in the context of numpy arrays.
```python
nums = range(5)    # range is a built-in function that creates a list of integers
print nums         # Prints "[0, 1, 2, 3, 4]"
print nums[2:4]    # Get a slice from index 2 to 4 (exclusive); prints "[2, 3]"
print nums[2:]     # Get a slice from index 2 to the end; prints "[2, 3, 4]"
print nums[:2]     # Get a slice from the start to index 2 (exclusive); prints "[0, 1]"
print nums[:]      # Get a slice of the whole list; prints ["0, 1, 2, 3, 4]"
print nums[:-1]    # Slice indices can be negative; prints ["0, 1, 2, 3]"
nums[2:4] = [8, 9] # Assign a new sublist to a slice
print nums         # Prints "[0, 1, 8, 8, 4]"
```

5. Loops: You can loop over the elements of a list like this(**Remember the colon**):
```python
animals = ['cat', 'dog', 'monkey']
for animal in animals:
    print animal
# Prints "cat", "dog", "monkey", each on its own line.
```
6. If you want access to the index of each element within the body of a loop, use the built-in enumerate function:
```python
animals = ['cat', 'dog', 'monkey']
for idx, animal in enumerate(animals):
    print '#%d: %s' % (idx + 1, animal)
# Prints "#1: cat", "#2: dog", "#3: monkey", each on its own line
```
7. List comprehensions: When programming, frequently we want to transform one type of data into another.
```python
nums = [1,2,3]
new_nums = nums * 2    #new_nums = [1,2,3,1,2,3]
t_nums = [x * 2 for x in nums] # t_nums = [2,4,6]
```
8. List comprehensions can also contain conditions:
```python
nums = [0, 1, 2, 3, 4]
even_squares = [x ** 2 for x in nums if x % 2 == 0]
print even_squares  # Prints "[0, 4, 16]"
```
9. Tuples: A tuple is an (immutable)(不能改变value) ordered list of values. A tuple is in many ways similar to a list; one of the most important differences is that tuples can be used as keys in dictionaries and as elements of sets, while lists cannot. Here is a trivial example:
```python
d = {(x, x + 1): x for x in range(10)}  # Create a dictionary with tuple keys
t = (5, 6)       # Create a tuple
print type(t)    # Prints "<type 'tuple'>"
print d[t]       # Prints "5"
print d[(1, 2)]  # Prints "1"
```
10. Dictionaries: A dictionary stores (key, value) pairs, similar to a Map in Java or an object in Javascript. You can use it like this
```python
d = {'cat': 'cute', 'dog': 'furry'}  # Create a new dictionary with some data
print d['cat']       # Get an entry from a dictionary; prints "cute"
print 'cat' in d     # Check if a dictionary has a given key; prints "True"
d['fish'] = 'wet'    # Set an entry in a dictionary
print d['fish']      # Prints "wet"
# print d['monkey']  # KeyError: 'monkey' not a key of d
print d.get('monkey', 'N/A')  # Get an element with a default; prints "N/A"
print d.get('fish', 'N/A')    # Get an element with a default; prints "wet"
del d['fish']        # Remove an element from a dictionary
print d.get('fish', 'N/A') # "fish" is no longer a key; prints "N/A"
```
	* It is easy to iterate over the keys in a dictionary: 
	```python
	d = {'person': 2, 'cat': 4, 'spider': 8}
	for animal in d:
	    legs = d[animal]
	    print 'A %s has %d legs' % (animal, legs)
	# Prints "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"
	```
	*  If you want access to keys and their corresponding values, use the iteritems method:
	```python
	d = {'person': 2, 'cat': 4, 'spider': 8}
	for animal, legs in d.iteritems():
	    print 'A %s has %d legs' % (animal, legs)
	# Prints "A person has 2 legs", "A spider has 8 legs", "A cat has 4 legs"
	```
	* Dictionary comprehensions:
	```python
	nums = [0, 1, 2, 3, 4]
	even_num_to_square = {x: x ** 2 for x in nums if x % 2 == 0}
	print even_num_to_square  # Prints "{0: 0, 2: 4, 4: 16}"
	```
11. Sets: A set is an unordered collection of distinct elements. As a simple example, consider the following:
```python
animals = {'cat', 'dog'}
print 'cat' in animals   # Check if an element is in a set; prints "True"
print 'fish' in animals  # prints "False"
animals.add('fish')      # Add an element to a set
print 'fish' in animals  # Prints "True"
print len(animals)       # Number of elements in a set; prints "3"
animals.add('cat')       # Adding an element that is already in the set does nothing
print len(animals)       # Prints "3"
animals.remove('cat')    # Remove an element from a set
print len(animals)       # Prints "2"
```
	* Loops: Iterating over a set has the same syntax as iterating over a list; however since sets are unordered, you cannot make assumptions about the order in which you visit the elements of the set:
	```python
	animals = {'cat', 'dog', 'fish'}
	for idx, animal in enumerate(animals):
	    print '#%d: %s' % (idx + 1, animal)
	# Prints "#1: fish", "#2: dog", "#3: cat"
	```
12. Functions: Using def keyword, no return value type and input type:
```python
def hello(name, loud=False):
    if loud:
        print 'HELLO, %s' % name.upper()
    else:
        print 'Hello, %s!' % name

hello('Bob') # Prints "Hello, Bob"
hello('Fred', loud=True)  # Prints "HELLO, FRED!"
```
13. Classes(usually need to to write the constructor function__init__)
```python
class Greeter:

    # Constructor
    def __init__(self, name):
        self.name = name  # Create an instance variable

    # Instance method
    def greet(self, loud=False):
        if loud:
            print 'HELLO, %s!' % self.name.upper()
        else:
            print 'Hello, %s' % self.name

g = Greeter('Fred')  # Construct an instance of the Greeter class
g.greet()            # Call an instance method; prints "Hello, Fred"
g.greet(loud=True)   # Call an instance method; prints "HELLO, FRED!"
```
### Numpy
1. Arrays
```python
import numpy as np

a = np.array([1, 2, 3])  # Create a rank 1 array
print type(a)            # Prints "<type 'numpy.ndarray'>"
print a.shape            # Prints "(3,)"
print a[0], a[1], a[2]   # Prints "1 2 3"
a[0] = 5                 # Change an element of the array
print a                  # Prints "[5, 2, 3]"

b = np.array([[1,2,3],[4,5,6]])   # Create a rank 2 array
print b.shape                     # Prints "(2, 3)"
print b[0, 0], b[0, 1], b[1, 0]   # Prints "1 2 4"
print b[0]                        #[1 2 3] which is different from Matlab
```
2. Numpy also provides many functions to create arrays:
```python
import numpy as np

a = np.zeros((2,2))  # Create an array of all zeros
print a              # Prints "[[ 0.  0.]
                     #          [ 0.  0.]]"

b = np.ones((1,2))   # Create an array of all ones
print b              # Prints "[[ 1.  1.]]"

c = np.full((2,2), 7) # Create a constant array
print c               # Prints "[[ 7.  7.]
                      #          [ 7.  7.]]"

d = np.eye(2)        # Create a 2x2 identity matrix
print d              # Prints "[[ 1.  0.]
                     #          [ 0.  1.]]"

e = np.random.random((2,2)) # Create an array filled with random values
print e                     # Might print "[[ 0.91940167  0.08143941]
                            #               [ 0.68744134  0.87236687]]"
```
3. indexing
```python
import numpy as np

# Create the following rank 2 array with shape (3, 4)
# [[ 1  2  3  4]
#  [ 5  6  7  8]
#  [ 9 10 11 12]]
a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])

# Use slicing to pull out the subarray consisting of the first 2 rows
# and columns 1 and 2; b is the following array of shape (2, 2):
# [[2 3]
#  [6 7]]
b = a[:2, 1:3] #This is just reference, if you want to copy, use np.copy(a[:2, 1:3])

# A slice of an array is a view into the same data, so modifying it
# will modify the original array.
print a[0, 1]   # Prints "2"
b[0, 0] = 77    # b[0, 0] is the same piece of data as a[0, 1]
print a[0, 1]   # Prints "77"
```
	* **Integer Indexing(very important for vectorization,different from MATLAB)**:
	```python
	import numpy as np

	a = np.array([[1,2], [3, 4], [5, 6]])

	# An example of integer array indexing.
	# The returned array will have shape (3,) and 
	print a[[0, 1, 2], [0, 1, 0]]  # Prints "[1 4 5]"

	# The above example of integer array indexing is equivalent to this:
	print np.array([a[0, 0], a[1, 1], a[2, 0]])  # Prints "[1 4 5]"


	# When using integer array indexing, you can reuse the same
	# element from the source array:
	print a[[0, 0], [1, 1]]  # Prints "[2 2]"

	# Equivalent to the previous integer array indexing example
	print np.array([a[0, 1], a[0, 1]])  # Prints "[2 2]"
	```
	When you index into numpy arrays using slicing, the resulting array view will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array. Here is an example:
	* Boolean array indexing lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition.
	```python
	import numpy as np

	a = np.array([[1,2], [3, 4], [5, 6]])

	bool_idx = (a > 2)  # Find the elements of a that are bigger than 2;
	                    # this returns a numpy array of Booleans of the same
	                    # shape as a, where each slot of bool_idx tells
	                    # whether that element of a is > 2.

	print bool_idx      # Prints "[[False False]
	                    #          [ True  True]
	                    #          [ True  True]]"

	# We use boolean array indexing to construct a rank 1 array
	# consisting of the elements of a corresponding to the True values
	# of bool_idx
	print a[bool_idx]  # Prints "[3 4 5 6]"

	# We can do all of the above in a single concise statement:
	print a[a > 2]     # Prints "[3 4 5 6]"

	```
4. Datatypes: 
```python
import numpy as np
x = np.array([1, 2])  # Let numpy choose the datatype
print x.dtype         # Prints "int64"
x = np.array([1.0, 2.0])  # Let numpy choose the datatype
print x.dtype             # Prints "float64"
x = np.array([1, 2], dtype=np.int64)  # Force a particular datatype
print x.dtype                         # Prints "int64
```
5. Array math (remember the difference between '*' and np.dot())
```python
import numpy as np
x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)
x+y  # same as np.add(x,y)

x-y  # same as np.substract(x,y)

x * y # same as np.multiply(x,y), Elementwise product

x.dot(y)  # same as np.dot(x,y), matrix multiplication

x/y   # same as np.divide(x,y), 
```
 	* Sum
 	```python
 	import numpy as np
	x = np.array([[1,2],[3,4]])
	print np.sum(x)  # Compute sum of all elements; prints "10"
	print np.sum(x, axis=0)  # Compute sum of each column; prints "[4 6]"
	print np.sum(x, axis=1)  # Compute sum of each row; prints "[3 7]"
	```
	* Transpose
	```python
	x = np.array([[1,2], [3,4]])
	print x    # Prints "[[1 2]
	           #          [3 4]]"
	print x.T  # Prints "[[1 3]
	           #          [2 4]]"

	# Note that taking the transpose of a rank 1 array does nothing:
	v = np.array([1,2,3])
	print v    # Prints "[1 2 3]"
	print v.T  # Prints "[1 2 3]"
	```
6. Broadcasting is a powerful mechanism that allows numpy to work with arrays of different shapes when performing arithmetic operations. Frequently we have a smaller array and a larger array, and we want to use the smaller array multiple times to perform some operation on the larger array.
One easiest way, we can do this 
```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
print vv                 # Prints "[[1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]
                         #          [1 0 1]]"
y = x + vv  # Add x and vv elementwise
print y  # Prints "[[ 2  2  4
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]
```
But with broadcasting, we can do this: 
```python
import numpy as np

# We will add the vector v to each row of the matrix x,
# storing the result in the matrix y
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # Add v to each row of x using broadcasting
print y  # Prints "[[ 2  2  4]
         #          [ 5  5  7]
         #          [ 8  8 10]
         #          [11 11 13]]"
```
###Scipy
1. Image Operation
```python
from scipy.misc import imread, imsave, imresize
# Read an JPEG image into a numpy array
img = imread('assets/cat.jpg')
print img.dtype, img.shape  # Prints "uint8 (400, 248, 3)"

# We can tint the image by scaling each of the color channels
# by a different scalar constant. The image has shape (400, 248, 3);
# we multiply it by the array [1, 0.95, 0.9] of shape (3,);
# numpy broadcasting means that this leaves the red channel unchanged,
# and multiplies the green and blue channels by 0.95 and 0.9
# respectively.
img_tinted = img * [1, 0.95, 0.9]

# Resize the tinted image to be 300 by 300 pixels.
img_tinted = imresize(img_tinted, (300, 300))

# Write the tinted image back to disk
imsave('assets/cat_tinted.jpg', img_tinted)
```
2. Matlab Files: The functions scipy.io.loadmat and scipy.io.savemat allow you to read and write MATLAB files.
3. Distance between points: The function scipy.spatial.distance.pdist computes the distance between all pairs of points in a given set:
```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
# Create the following array where each row is a point in 2D space:
# [[0 1]
#  [1 0]
#  [2 0]]
x = np.array([[0, 1], [1, 0], [2, 0]])
print x

# Compute the Euclidean distance between all rows of x.
# d[i, j] is the Euclidean distance between x[i, :] and x[j, :],
# and d is the following array:
# [[ 0.          1.41421356  2.23606798]
#  [ 1.41421356  0.          1.        ]
#  [ 2.23606798  1.          0.        ]]
d = squareform(pdist(x, 'euclidean'))
print d
```

###Maplotlib
* Plotting
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
* Subplots
```python
import numpy as np
import matplotlib.pyplot as plt

# Compute the x and y coordinates for points on sine and cosine curves
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

