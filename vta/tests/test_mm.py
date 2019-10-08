import numpy as np

#A = np.arange((7 * 23)).reshape((7, 23))
#B = np.arange((23 * 19)).reshape((23, 19))
A = np.arange((7 * 23)).reshape((7,23))
B = np.arange((19 * 23)).reshape((19,23)).transpose(1,0)

print (A)
print (B)
C = np.matmul(A, B)

print (C)
