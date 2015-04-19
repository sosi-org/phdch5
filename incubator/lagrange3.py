
import numpy as np
import scipy
import matcompat

import matplotlib.pylab as plt

def lagrange3(x, y, xx):

    # Local Variables: y, x, px, xx
    # Function calls: lagrange3
    #px=(xx-x(2))*(xx-x(3))/((x(1)-x(2))*(x(1)-x(3)))*y(1)+(xx-x(1))*(xx-x(3))/((x(2)-x(1))*(x(2)-x(3)))*y(2)+(xx-x(1))*(xx-x(2))/((x(3)-x(1))*(x(3)-x(2)))*y(3);
    #px = np.dot(matdiv(np.dot(xx-x[1], xx-x[2]), np.dot(x[0]-x[1], x[0]-x[2])), y[0])+np.dot(matdiv(np.dot(xx-x[0], xx-x[2]), np.dot(x[1]-x[0], x[1]-x[2])), y[1])+np.dot(matdiv(np.dot(xx-x[0], xx-x[1]), np.dot(x[2]-x[0], x[2]-x[1])), y[2])
    A=np.dot(xx-x[1], xx-x[2])
    B=np.dot(x[0]-x[1], x[0]-x[2])
    C=np.dot(xx-x[0], xx-x[2])
    D=np.dot(x[1]-x[0], x[1]-x[2])
    E=np.dot(xx-x[0], xx-x[1])
    F=np.dot(x[2]-x[0], x[2]-x[1])
    #px = np.dot(matdiv(A / B), y[0])+np.dot(matdiv(C, D), y[1])+np.dot(matdiv(E, F), y[2])
    px = (A / B) * y[0] +   (C/ D) * y[1] + (E / F) * y[2]
    return px
