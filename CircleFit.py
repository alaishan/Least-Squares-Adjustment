#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11 Mar 20 13:52:14 2020

@author: Alaisha Naidu
Name: General Least Squares for Circle of Best Fit
"""

import math
from math import sqrt, pi
import numpy as np
import sympy as sp
from sympy import Matrix, symbols, atan, sqrt
NDXALA004 APG4005F – Tutorial 1
 Xb, Xc, Xd, Xe, Yb, Yc, Yd, Ye = symbols('Xb Xc Xd Xe Yb Yc Yd Ye', real = True)
#observation equations
l1 = sp.atan((Xd-Xc)/(Yd-Yc)) + pi - sp.atan((Xb-Xc)/(Yb-Yc)) l2 = sp.atan((Xe-Xd)/(Ye-Yd)) + pi - sp.atan((Xc-Xd)/(Yc-Yd)) l3 = sp.atan((Xe-Xd)/(Ye-Yd))
l4 = sp.atan((Xc-Xb)/(Yc-Yb))
l5 = sp.sqrt(((Xc-Xb)**2) + ((Yc-Yb)**2))
l6 = sp.sqrt(((Xd-Xc)**2) + ((Yd-Yc)**2))
l7 = sp.sqrt(((Xe-Xd)**2) + ((Ye-Yd)**2))
x = sp.Matrix([[l1], [l2], [l3], [l4], [l5], [l6], [l7]])
y = sp.Matrix([[Xc], [Xd], [Yc], [Yd]]) # Coordinates for C and D are not fixed
z = sp.Matrix([[Xb], [Xe], [Yb], [Ye]]) # Coordinates for B and E are fixed
A1 = x.jacobian(y)
obs = [(Xb,1000), (Yb,1000), (Xc,1173), (Yc,1100), (Xd,1223), (Yd,1186), (Xe,1400), (Ye,1186.5)]
A = A1.subs(obs)
fx0 = x.subs(obs)# Derived observations
Fx0 = np.array(fx0).astype(np.float64)
print(Fx0)
l= sp.Matrix([[2.617921156],[4.189081093],[1.570796327],[1.046979385],[199.880 ],[99.900],[177.000]])
Y = y.subs(obs)
L = z.subs(obs)
w = fx0 - l
P = np.diag([(1/0.00004848)**2, (1/0.00004848)**2, (1/0.000009693)**2, (1/0.000009693)**2, (1/0.003399)**2, (1/0.005)**2, (1/0.005)**2])
N = A.T*P*A
n = np.array(N).astype(np.float64) #convert N into a numpy array/matrix from a sympy matrix
NIn = np.linalg.inv(n) #invert N
u = A.T*P*w
sigma = -NIn*u
final_coords = Y + sigma
fc = np.array(final_coords).astype(np.float64) print(fc)