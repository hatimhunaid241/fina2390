import numpy as np

np.random.seed(12345)

# choose the precision of the float numbers
np.set_printoptions(precision=4, suppress=True)

import timeit
import matplotlib.pyplot as plt

# set the figure size
plt.rc("figure", figsize=(8, 4))

# generate a seq of numbers in list
my_arr = np.arange(1000)

# get normal distr sample (rows, cols)
data = np.random.randn(2, 3)

# element wise operation - easier without loops
mul_data = data * 10
add_data = data + data

# get dimension/shape
shape_data = data.shape

# get no. of dimension/shape
shape_data = data.ndim

# get data type of elements
type_data = data.dtype

# create np data array (ndarray)
data1 = [6, 7.5, 8, 0, 1]
arr1 = np.array(data1)

# create nested np data array (ndarray)
data2 = [[1, 2, 3, 4], [5, 6, 7, 8]]
arr2 = np.array(data2)

# more ways to create pre-filled arrays
data_np_zeros = np.zeros(10)  # zeros
data_np_zeros_1 = np.zeros((3, 6))  # zeros
data_np_ones = np.ones((3, 2))  # ones
data_np_empty = np.empty((2, 3, 2))  # garbage (very lowest number > 0)

# define data types for array
arr1 = np.array([1, 2, 3], dtype=np.float64)

# convert d type
int_arr = arr1.astype(np.int32)  # dec truncated

# elementwise operations
arr2 = np.array([[0.0, 4.0, 1.0], [7.0, 2.0, 12.0]])
print(arr2)
print(arr2 > arr1)

# assigning multiple scalar values
arr2[1:2] = 1.0
arr2[:] = 64  # assign all

# copy for immutabillity
arr3 = arr2.copy()

# get multidimentional elements
arr2d = arr2
