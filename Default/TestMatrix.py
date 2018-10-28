import numpy as np

#转置
# arr = np.arange(6).reshape((3,2))
# print(arr)
# print(arr.T)

data = [[1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8,9]]
#data = [[1,2,3],[3,2,1]]
arr = np.array(data)
arr2 = arr.T

print("矩阵1")
print(arr2)
print("矩阵2")
print(arr)
print("=============矩阵1 * 矩阵2=============")
#矩阵相乘
arr3 = np.dot(arr2,arr)
print(arr3)