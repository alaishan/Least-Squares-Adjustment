#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 17:49:02 2020

@author: Alaisha Naidu
Name: Least Squares Adjustment Circle of Best Fit Cont.
"""
# Least Squares Adjustment - circle fit # NDXALA004
# 22 Feb 2020

import numpy as np
import sympy as sp
from sympy import symbols, Matrix

x0, x1, x2, x3, y0, y1, y2, y3, r0 = symbols('x0 x1 x2 x3 y0 y1 y2 y3 r0', real = True)
#observation equations
a1 = (x1-x0)**2 + (y1-y0)**2 - r0**2 
a2 = (x2-x0)**2 + (y2-y0)**2 - r0**2 
a3 = (x3-x0)**2 + (y3-y0)**2 - r0**2
x = sp.Matrix([[a1], [a2], [a3]])
y = sp.Matrix([[x0], [y0]])
z = sp.Matrix([[x1], [y1], [x2], [y2], [x3], [y3], [r0]])
A1 = x.jacobian(y) 
B1 = x.jacobian(z)
A = A1.subs({x0:491601.00, x1:491573.24, x2:491587.97, x3:491617.65, y0:5454931.00, y1:5454923.11, y2:5454956.30, y3:5454954.27, r0:28.88}) 
B = B1.subs({x0:491601.00, x1:491573.24, x2:491587.97, x3:491617.65, y0:5454931.00, y1:5454923.11, y2:5454956.30, y3:5454954.27, r0:28.88}) 
w = x.subs({x0:491601.00, x1:491573.24, x2:491587.97, x3:491617.65, y0:5454931.00, y1:5454923.11, y2:5454956.30, y3:5454954.27, r0:28.88}) 
Y = y.subs({x0:491601.00, y0:5454931.00})
l = z.subs({x1:491573.24, x2:491587.97, x3:491617.65, y1:5454923.11, y2:5454956.30, y3:5454954.27, r0:22.88})
Q = np.diag([0.01**2, 0.01**2, 0.01**2, 0.01**2, 0.01**2, 0.01**2, 0.01**2])
M = B*Q*B.T
m = np.array(M).astype(np.float64) #convert M into a numpy array/matrix from a sympy matrix
MIn = np.linalg.inv(m) #invert M
N = A.T*MIn*A
n = np.array(N).astype(np.float64) #convert N into a numpy array/matrix from a sympy matrix
NIn = np.linalg.inv(n) #invert N
# calculating the centre of the
u = A.T*MIn*w
sigma = -NIn*u
centre = Y + sigma
print("Centre of Circle: (x0,y0) =", centre)
