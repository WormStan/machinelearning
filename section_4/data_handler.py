#
# Numpy demo

import numpy as np

myarray = np.array([1,2,3])
print(myarray)
print(myarray.shape)

myarray = np.array([[1,2,3],[2,3,4],[4,5,6]])
print(myarray)
print(myarray.shape)


#
# Matplotlib demo

import matplotlib.pyplot as plt
import numpy as np

# myarray = np.array([[1,2,3],[2,3,4],[3,4,5]])

# plt.plot(myarray)
# plt.xlabel('x axis')
# plt.ylabel('y axis')
# plt.show()

myarray1 = np.array([1,2,3])
myarray2 = np.array([11,21,31])

plt.scatter(myarray1, myarray2)

plt.xlabel('x axis')
plt.ylabel('y axis')

plt.show()