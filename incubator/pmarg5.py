
import numpy as np
import scipy
import matcompat

import matplotlib.pylab as plt

def pmarg5(p, id, iu, L):

    # Local Variables: A, pp, B, d, pms, I, I0, j, iu, L, p, u, id, pm
    # Function calls: reshape, sum, ones, zeros, pmarg5
    #%pass to 0:L-1 representation
    #%u=L-id;%bit's order is reverse of bin's order !!!
    #%d=L-iu;
    u = iu-1.
    d = id-1.
    A = np.dot(np.ones((2.**(L-u-1.)), 1.), np.array(np.hstack((np.arange(0., (2.**d-1.)+1)))))
    B = np.dot(np.dot(2.**(u+1.), np.array(np.hstack((np.arange(0., (2.**(L-u-1.)-1.)+1)))).conj().T), np.ones(1., (2.**d)))
    I0 = 1.+np.reshape((A+B), np.array([]), 1.)
    pm = np.zeros((1., (2.**L)))
    pms = np.zeros((1., (2.**(u-d))))
    for j in np.arange(0., (2.**(u-d+1.)-1.)+1):
        I = I0+np.dot(2.**d, j)
        pp = np.sum(p[int(I)-1])
        pm[int(I)-1] = pp
        pms[int((j+1.))-1] = pp
        
    return [pm, pms]