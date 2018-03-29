"""
>>> import numpy as np
>>> a = np.array([1, 4, 5, 8], float)
>>> a
array([ 1., 4., 5., 8.])
>>> type(a)
<type 'numpy.ndarray'>

>>> a[:2]
array([ 1., 4.])
>>> a[3]
8.0
>>> a[0] = 5.
>>> a
array([ 5., 4., 5., 8.])

>>> a = np.array([[1, 2, 3], [4, 5, 6]], float)
>>> a
array([[ 1., 2., 3.],
[ 4., 5., 6.]])
>>> a[0,0]
1.0
>>> a[0,1]
2.0

>>> a = np.array([[1, 2, 3], [4, 5, 6]], float)
>>> a[1,:]
array([ 4., 5., 6.])
>>> a[:,2]
array([ 3., 6.])
>>> a[-1:,-2:]
array([[ 5., 6.]])

>>> a.shape
(2, 3)

>>> a.dtype
dtype('float64')

>>> a = np.array([[1, 2, 3], [4, 5, 6]], float)
>>> len(a)
2

>>> a = np.array([[1, 2, 3], [4, 5, 6]], float)
>>> 2 in a
True
>>> 0 in a
False

>>> a = np.array(range(10), float)
>>> a
array([ 0., 1., 2., 3., 4., 5., 6., 7., 8., 9.])
>>> a = a.reshape((5, 2))
>>> a
array([[ 0., 1.],
[ 2., 3.],
[ 4., 5.],
[ 6., 7.],
[ 8., 9.]])
>>> a.shape
(5, 2)

>>> a = np.array([1, 2, 3], float)
>>> b = a
>>> c = a.copy()
>>> a[0] = 0
>>> a
array([0., 2., 3.])
>>> b
array([0., 2., 3.])
>>> c
array([1., 2., 3.])

>>> a = np.array([1, 2, 3], float)
>>> a.tolist()
[1.0, 2.0, 3.0]
>>> list(a)
[1.0, 2.0, 3.0]

>>> a = array([1, 2, 3], float)
>>> s = a.tostring()
>>> s
'\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00
\x00\x00\x08@'
>>> np.fromstring(s)
array([ 1., 2., 3.])

>>> a = array([1, 2, 3], float)
>>> a
array([ 1., 2., 3.])
>>> a.fill(0)
>>> a
array([ 0., 0., 0.])

>>> a = np.array(range(6), float).reshape((2, 3))
>>> a
array([[ 0., 1., 2.],
[ 3., 4., 5.]])
>>> a.transpose()
array([[ 0., 3.],
[ 1., 4.],
[ 2., 5.]])

>>> a = np.array([[1, 2, 3], [4, 5, 6]], float)
>>> a
array([[ 1., 2., 3.],
[ 4., 5., 6.]])
>>> a.flatten()
array([ 1., 2., 3., 4., 5., 6.])

>>> a = np.array([1,2], float)
>>> b = np.array([3,4,5,6], float)
>>> c = np.array([7,8,9], float)
>>> np.concatenate((a, b, c))
array([1., 2., 3., 4., 5., 6., 7., 8., 9.])

>>> a = np.array([[1, 2], [3, 4]], float)
>>> b = np.array([[5, 6], [7,8]], float)
>>> np.concatenate((a,b))
array([[ 1., 2.],
[ 3., 4.],
[ 5., 6.],
[ 7., 8.]])
>>> np.concatenate((a,b), axis=0)
array([[ 1., 2.],
[ 3., 4.],
[ 5., 6.],
[ 7., 8.]])
>>> np.concatenate((a,b), axis=1)
array([[ 1., 2., 5., 6.],
[ 3., 4., 7., 8.]])

>>> a = np.array([1, 2, 3], float)
>>> a
array([1., 2., 3.])
>>> a[:,np.newaxis]
array([[ 1.],
[ 2.],
[ 3.]])
>>> a[:,np.newaxis].shape
(3,1)
>>> b[np.newaxis,:]
array([[ 1., 2., 3.]])
>>> b[np.newaxis,:].shape
(1,3)

>>> np.arange(5, dtype=float)
array([ 0., 1., 2., 3., 4.])
>>> np.arange(1, 6, 2, dtype=int)
array([1, 3, 5])

>>> np.ones((2,3), dtype=float)
array([[ 1., 1., 1.],
[ 1., 1., 1.]])
>>> np.zeros(7, dtype=int)
array([0, 0, 0, 0, 0, 0, 0])

>>> a = np.array([[1, 2, 3], [4, 5, 6]], float)
>>> np.zeros_like(a)
array([[ 0., 0., 0.],
[ 0., 0., 0.]])
>>> np.ones_like(a)
array([[ 1., 1., 1.],
[ 1., 1., 1.]])

>>> np.identity(4, dtype=float)
array([[ 1., 0., 0., 0.],
[ 0., 1., 0., 0.],
[ 0., 0., 1., 0.],
[ 0., 0., 0., 1.]])

>>> np.eye(4, k=1, dtype=float)
array([[ 0., 1., 0., 0.],
[ 0., 0., 1., 0.],
[ 0., 0., 0., 1.],
[ 0., 0., 0., 0.]])

>>> a = np.array([1,2,3], float)
>>> b = np.array([5,2,6], float)
>>> a + b
array([6., 4., 9.])
>>> a - b
array([-4., 0., -3.])
>>> a * b
array([5., 4., 18.])
>>> b / a
array([5., 1., 2.])
>>> a % b
array([1., 0., 3.])
>>> b**a
array([5., 4., 216.])

>>> a = np.array([[1,2], [3,4]], float)
>>> b = np.array([[2,0], [1,3]], float)
# multiplication remains elementwise.
>>> a * b
array([[2., 0.], [3., 12.]])

>>> a = np.array([1,2,3], float)
>>> b = np.array([4,5], float)
>>> a + b
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
ValueError: shape mismatch: objects cannot be broadcast to a single shape

>>> a = np.array([[1, 2], [3, 4], [5, 6]], float)
>>> b = np.array([-1, 3], float)
>>> a
array([[ 1., 2.],
[ 3., 4.],
[ 5., 6.]])
>>> b
array([-1., 3.])
>>> a + b
array([[ 0., 5.],
[ 2., 7.],
[ 4., 9.]])

>>> a = np.zeros((2,2), float)
>>> b = np.array([-1., 3.], float)
>>> a
array([[ 0., 0.],
[ 0., 0.]])
>>> b
array([-1., 3.])
>>> a + b
array([[-1., 3.],
[-1., 3.]])
>>> a + b[np.newaxis,:]
array([[-1., 3.],
[-1., 3.]])
>>> a + b[:,np.newaxis]
array([[-1., -1.],
[ 3., 3.]])

>>> a = np.array([1, 4, 9], float)
>>> np.sqrt(a)
array([ 1., 2., 3.])

>>> a = np.array([1.1, 1.5, 1.9], float)
# The functions floor, ceil, and rint give the lower, upper, or nearest (rounded) integer
>>> np.floor(a)
array([ 1., 1., 1.])
>>> np.ceil(a)
array([ 2., 2., 2.])
>>> np.rint(a)
array([ 1., 2., 2.])

>>> np.pi
3.1415926535897931
>>> np.e
2.7182818284590451

>>> a = np.array([1, 4, 5], int)
>>> for x in a:
        print x
    <hit return>
1 4 5

>>> a = np.array([[1, 2], [3, 4], [5, 6]], float)
>>> for x in a:
        print x
    <hit return>
[ 1. 2.]
[ 3. 4.]
[ 5. 6.]

>>> a = np.array([[1, 2], [3, 4], [5, 6]], float)
>>> for (x, y) in a:
        print x * y
    <hit return>
2.0
12.0
30.0

>>> a = np.array([2, 4, 3], float)
>>> a.sum()
9.0
>>> a.prod()
24.0
>>> np.sum(a)
9.0
>>> np.prod(a)
24.0

>>> a = np.array([2, 1, 9], float)
# mean (average), variance, and standard deviation
>>> a.mean()
4.0
>>> a.var()
12.666666666666666
>>> a.std()
3.5590260840104371

>>> a = np.array([2, 1, 9], float)
>>> a.min()
1.0
>>> a.max()
9.0

>>> a = np.array([2, 1, 9], float)
# The argmin and argmax functions return the array indices of the minimum and maximum values.
>>> a.argmin()
1
>>> a.argmax()
2

>>> a = np.array([[0, 2], [3, -1], [3, 5]], float)
>>> a.mean(axis=0)
array([ 2., 2.])
>>> a.mean(axis=1)
array([ 1., 1., 4.])
>>> a.min(axis=1)
array([ 0., -1., 3.])
>>> a.max(axis=0)
array([ 3., 5.])

>>> a = np.array([6, 2, 5, -1, 0], float)
>>> sorted(a)
[-1.0, 0.0, 2.0, 5.0, 6.0]
>>> a.sort()
>>> a
array([-1., 0., 2., 5., 6.])

>>> a = np.array([6, 2, 5, -1, 0], float)
# Return an array whose values are limited to ``[min, max]``.
>>> a.clip(0, 5)
array([ 5., 2., 5., 0., 0.])

>>> a = np.array([1, 1, 4, 5, 5, 5, 7], float)
>>> np.unique(a)
array([ 1., 4., 5., 7.])

>>> a = np.array([[1, 2], [3, 4]], float)
>>> a.diagonal()
array([ 1., 4.])

>>> a = np.array([1, 3, 0], float)
>>> b = np.array([0, 3, 2], float)
>>> a > b
array([ True, False, False], dtype=bool)
>>> a == b
array([False, True, False], dtype=bool)
>>> a <= b
array([False, True, True], dtype=bool)

>>> c = a > b
>>> c
array([ True, False, False], dtype=bool)

>>> a = np.array([1, 3, 0], float)
>>> a > 2
array([False, True, False], dtype=bool)

>>> c = np.array([ True, False, False], bool)
# whether or not any or all elements of a Boolean array are true
>>> any(c)
True
>>> all(c)
False

>>> a = np.array([1, 3, 0], float)
>>> np.logical_and(a > 0, a < 3)
array([ True, False, False], dtype=bool)
>>> b = np.array([True, False, True], bool)
>>> np.logical_not(b)
array([False, True, False], dtype=bool)
>>> c = np.array([False, True, False], bool)
>>> np.logical_or(b, c)
array([ True, True, False], dtype=bool)

>>> a = np.array([1, 3, 0], float)
>>> np.where(a != 0, 1 / a, a)
array([ 1. , 0.33333333, 0. ])
>>> np.where(a > 0, 3, 2)
array([3, 3, 2])

>>> a = np.array([[0, 1], [3, 0]], float)
>>> a.nonzero()
(array([0, 1]), array([1, 0]))

>>> a = np.array([1, np.NaN, np.Inf], float)
>>> a
array([ 1., NaN, Inf])
>>> np.isnan(a)
array([False, True, False], dtype=bool)
>>> np.isfinite(a)
array([ True, False, False], dtype=bool)

>>> a = np.array([[6, 4], [5, 9]], float)
>>> a >= 6
array([[ True, False],
[False, True]], dtype=bool)
>>> a[a >= 6]
# an array with only the True elements is returned
array([ 6., 9.])

>>> a = np.array([[6, 4], [5, 9]], float)
>>> sel = (a >= 6)
>>> a[sel]
array([ 6., 9.])

>>> a[np.logical_and(a > 5, a < 9)]
>>> array([ 6.])

>>> a = np.array([2, 4, 6, 8], float)
>>> b = np.array([0, 0, 1, 3, 2, 1], int)
# the integer arrays contain the indices of the elements to be taken from an array.
>>> a[b]
array([ 2., 2., 4., 8., 6., 4.])

>>> a = np.array([2, 4, 6, 8], float)
>>> a[[0, 0, 1, 3, 2, 1]]
array([ 2., 2., 4., 8., 6., 4.])

>>> a = np.array([[1, 4], [9, 16]], float)
>>> b = np.array([0, 0, 1, 1, 0], int)
>>> c = np.array([0, 1, 1, 1, 1], int)
>>> a[b,c]
array([ 1., 4., 16., 16., 4.])

>>> a = np.array([2, 4, 6, 8], float)
>>> b = np.array([0, 0, 1, 3, 2, 1], int)
'''
the first element taken has a first axis index taken from the first member of the first
selection array, a second index from the first member of the second selection array, and so on.
An example
a.take(b) == a[b]
'''
>>> a.take(b)
array([ 2., 2., 4., 8., 6., 4.])

>>> a = np.array([[0, 1], [2, 3]], float)
>>> b = np.array([0, 0, 1], int)
>>> a.take(b, axis=0)
array([[ 0., 1.],
[ 0., 1.],
[ 2., 3.]])
>>> a.take(b, axis=1)
array([[ 0., 0., 1.],
[ 2., 2., 3.]])

>>> a = np.array([0, 1, 2, 3, 4, 5], float)
>>> b = np.array([9, 8, 7], float)
# Put function will take values from a source array and place them at specified indices in the array calling put
>>> a.put([0, 3], b)
>>> a
array([ 9., 1., 2., 8., 4., 5.])

>>> a = np.array([0, 1, 2, 3, 4, 5], float)
>>> a.put([0, 3], 5)
>>> a
array([ 5., 1., 2., 5., 4., 5.])

>>> a = np.array([1, 2, 3], float)
>>> b = np.array([0, 1, 1], float)
# vector and matrix multiplication
>>> np.dot(a, b)
5.0

>>> a = np.array([[0, 1], [2, 3]], float)
>>> b = np.array([2, 3], float)
>>> c = np.array([[1, 1], [4, 0]], float)
>>> a
array([[ 0., 1.],
[ 2., 3.]])
>>> np.dot(b, a)
array([ 6., 11.])
>>> np.dot(a, b)
array([ 3., 13.])
>>> np.dot(a, c)
array([[ 4., 0.],
[ 14., 2.]])
>>> np.dot(c, a)
array([[ 2., 4.],
[ 0., 4.]])

>>> a = np.array([1, 4, 0], float)
>>> b = np.array([2, 2, 1], float)
>>> np.outer(a, b)
array([[ 2., 2., 1.],
[ 8., 8., 4.],
[ 0., 0., 0.]])
>>> np.inner(a, b)
10.0
>>> np.cross(a, b)
array([ 4., -1., -6.])


"""