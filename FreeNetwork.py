#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 05 13:11:54 2020

@author: Alaisha Naidu
Name: Free Network Adjustment
Creds: University of Cape Town
"""

# 5 May 2020
# APG4005F - Tutorial 5 
# NDXALA004

import math 
from math import sqrt, pi
import numpy as np
import sympy as sp
from sympy import Matrix, symbols, atan

ZA, Z1, Z2, Z3 = symbols('ZA Z1 Z2 Z3', real = True)

#observation equations
l1 = Z1 - ZA
l2 = Z2 - Z1
l3 = Z3 - Z2
l4 = ZA - Z3
l5 = Z2 - ZA


l = sp.Matrix([[l1], [l2], [l3], [l4], [l5]])
x = sp.Matrix([[ZA], [Z1], [Z2], [Z3]]) # heights of each point in the network

A1 = l.jacobian(x)

H = [(ZA, 214.880), (Z1, 276.358), (Z2, 293.352), (Z3, 268.301)] #elevation of points
L = sp.Matrix([[61.478], [16.994], [-25.051], [-53.437], [78.465]]) #observed height differences 

ht = x.subs(H)
A = A1.subs(H)
fx0 = l.subs(H)
w = fx0 - L

P = np.diag([(1/0.005)**2, (1/0.01)**2, (1/0.005)**2, (1/0.01)**2, (1/0.01)**2])

G = sp.Matrix([[1], [1], [1], [1]]) #singular matrix because no height is held fixed

u = A.T*P*w
v = A.T*P*A
t = A.T*P*A + G*G.T

T = np.array(t).astype(np.float64) #convert t into a numpy array/matrix from a sympy matrix
TIn = np.linalg.inv(T) #invert T

sigma = -TIn*v*TIn*u
final_height = ht + sigma
fh = np.array(final_height).astype(np.float64)
print(fh)