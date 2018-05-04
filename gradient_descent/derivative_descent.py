#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def problem(x):
    e = 2.71828182845904590
    return x**3 + 2*x + e**x - 3

def error(x):
    return (problem(x)-0)**2

def derivative_descent(x):
    delta = 0.00000001
    derivative = (error(x + delta) - error(x - delta)) / (delta * 2)
    alpha = 0.01
    x = x - derivative * alpha
    return x
x = 1.0
for i in range(50):
    x = derivative_descent(x)
    y = format(problem(x),'0.6f')
    print('x = {:6f}, problem(x) = {:6f},error(x) = {:6f}'.format(x,problem(x),error(x)))

x = np.linspace(0,1,100)
plt.plot(x,error(x))
plt.plot(x,problem(x))
plt.show()
