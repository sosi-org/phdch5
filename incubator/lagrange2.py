
import numpy as np
#import scipy
#import matcompat
#import matplotlib.pylab as plt


def matdiv(a,b):
   return a/b

def lagrange2(x, y, xx):

    # Local Variables: y, x, px, xx
    # Function calls: lagrange2
    px = np.dot(matdiv(xx-x[1], x[0]-x[1]), y[0])+np.dot(matdiv(xx-x[0], x[1]-x[0]), y[1])
    return [px]