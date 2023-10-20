# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 18:57:15 2023

@author: Lucien C
"""
import numpy as np
vect  = np.linspace(0,20,21)
print(vect)
vect [9:16] = -vect[9:16]
print(vect)


exo2 = np.linspace(15,55,10)
print(exo2)
print(exo2[1:-1])

exo3 = np.array([[3,1,2,3],[3,3,3,3],[3,3,3,3],[3,3,3,3]])
print(exo3)
for i in range(0,exo3.shape[0]):
    for j in range(0,exo3.shape[1]):
        print(exo3[i,j])
    
exo4 = np.linspace(5,50,10)
print(exo4)

exo5 = np.linspace(0,10,5)
print(exo5)

vect_1 = np.array([[1,1],[2,2]])
vect_2 = np.zeros((2,2))
vect_3 = vect_1*vect_2
print(vect_3)

exo7 = np.array([[np.linspace(10,21,4)],[np.linspace(10,21,4)],[np.linspace(10,21,4)]])
print(exo7)


#exo8 matrix.shape(0 or 1)

exo9 = np.zeros((4,4))
print(exo9)
exo9[::2,1::2] = 1
exo9[1::2,::2] = 1
print(exo9)


a = np.array([0,10,20,40,60])
b = np.array([10,30,40])
print(np.intersect1d(a,b))


exo11 = np.array([10, 10, 20, 20, 30, 30])


print(np.unique(exo11))
exo11 = np.array([[1, 1], [2, 3]])

print(np.unique(exo11))


exo12_1 = [[1, 0], [0, 1]]
exo12_2 = [[1, 2], [3, 4]]

print(exo12_1)
print(exo12_2)
result1 = np.cross(exo12_1, exo12_2)
result2 = np.cross(exo12_2, exo12_1)

print(result1)

print(result2)


exo13= np.random.random((10,2))
x,y = exo13[:,0], exo13[:,1]
r = np.sqrt(x**2+y**2)
t = np.arctan2(y,x)
print(r)
print(t)

exo14 = np.arange(100)
print(exo14)
a = np.random.uniform(0,100)
print(a)
index = (np.abs(exo14-a)).argmin()
print(exo14[index])

import numpy as np
test = np.linspace(0,10,15).reshape(3,5)

print(test)
