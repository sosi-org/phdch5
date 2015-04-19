
import numpy as np
#import scipy
#import matcompat

import matplotlib.pylab as plt
from consts import EPS


def hmarg(pn, pd, L):
    #%it computes te entropy by summing over the marginal entropies
    #%pn and pd contain the marginals

    # Local Variables: i, h, pn, L, pd
    # Function calls: hmarg, log2, sum, eps
    h = -np.sum((pn[0,:]*np.log2((pn[0,:]+EPS))))
    for i in np.arange(2., (L)+1):
        h = h-np.sum((pn[int(i)-1,:]*np.log2((pn[int(i)-1,:]+EPS))))+np.sum((pd[int(i)-1,:]*np.log2((pd[int(i)-1,:]+EPS))))
        
    #%h=h;
    #print type(h)
    #assert type(h) is np.float64
    return h