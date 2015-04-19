
import numpy as np
#import scipy
#import matcompat

import matplotlib.pylab as plt

#%inner function
def regular_discretize(y, M, ds, y0):

    # Local Variables: y, y0, M, ds, z2
    # Function calls: size, zeros, floor, regular_discretize
    sz=y.shape
    z2 = np.zeros([y.size], int) #np.zeros(y.shape)
    #z2[:] = np.floor(matdiv(y.flatten(1)-y0, ds))+1.0.
    z2 = np.floor(((y.flatten()-y0)/ ds))+1
    #%edges(J) ... +J-1
    #%out of bounds: oob = sum(spk2(:)>M) + sum((spk2(:)<0));
    #z2[:] = z2.flatten(1)*(z2.flatten(1) > 0.)
    #z2[:] = z2.flatten(1)+1.
    z2 = z2*(z2 > 0) #and
    z2 = z2+1
    #%range: 1:M
    #z2[:] = np.dot(z2.flatten(1) > M, M)+(z2.flatten(1)<=M)*z2.flatten(1)
    z2 = ((z2 > M)* M)+(z2<=M)*z2
    print "Warning: not tested"
    return z2.reshape(sz)
