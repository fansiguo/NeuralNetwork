#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def problem(x):
    e = 2.71828182845904590
    return x[0]**5 + e**x[1] + x[0]**3 + x[0] + x[1] - 5

def error(x):
    return (problem(x)-0)**2

def gradient_descent(x):
    delta = 0.00000001

    derivative_x0 = (error([x[0] + delta, x[1]]) - error([x[0] - delta, x[1]])) / (delta * 2)
    derivative_x1 = (error([x[0], x[1] + delta]) - error([x[0], x[1] - delta])) / (delta * 2)

    alpha = 0.01
    x[0] = x[0] - derivative_x0 * alpha
    x[1] = x[1] - derivative_x1 * alpha
    return [x[0],x[1]]

x = [0.0, 0.0]
for i in range(50):
    x = gradient_descent(x)
    print('x = {:6f},{:6f}, problem(x) = {:6f}'.format(x[0],x[1],problem(x)))