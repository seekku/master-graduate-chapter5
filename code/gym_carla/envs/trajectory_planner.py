import math
import time

import matplotlib.pyplot as plt
import numpy as np

A = np.array([[2,1,-2],[3,0,1],[1,1,-1]])
b = np.array([[-3],[5],[-2]])

before = time.time()
x = np.linalg.solve(A,b)
print(x)
after = time.time()
print('time:',after-before)  #demo for Ax = b 求取x


#parameter
init_s = 0
end_s = 6

Delta = 11


def solve_matrix(l0,l1):
    S = np.array([[init_s**5,init_s**4,init_s**3,init_s**2,init_s,1],
                  [5*init_s**4,4*init_s**3,3*init_s**2,2*init_s,1,0],
                  [20*init_s**3,12*init_s**2,6*init_s,2,0,0],
                  [end_s**5,end_s**4,end_s**3,end_s**2,end_s,1],
                  [5*end_s**4,4*end_s**3,3*end_s**2,2*end_s,1,0],
                  [20*end_s**3,12*end_s**2,6*end_s,2,0,0]])
    l = np.array([[l0],[0],[0],[l1],[0],[0]])  #l的偏移
    a = np.linalg.solve(S,l)        #权重系数

    return a


def calculate_point(a):
    s_list = []
    l_list = []
    for i in range(Delta):
        s = init_s + i*(end_s-init_s)/((Delta-1)*1.00)
        l = a[0]*s**5+a[1]*s**4+a[2]*s**3+a[3]*s**2+a[4]*s+a[5]
        s_list.append(s)
        l_list.append(l)
    return s_list,l_list





