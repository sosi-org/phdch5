
import numpy as np
import scipy
import matcompat
#from CON*
import matplotlib.pylab as plt
from consts import EPS

from pmarg4 import pmarg4

def probqrs(p, L, q, M):
    assert type(q) is int
    # Local Variables: p, i, M, L, q, pqq, il, pd, pn, pm
    # Function calls: probqrs, pmarg4, zeros, eps, size
    #pm = np.zeros(matcompat.size(p),int)
    pm = np.zeros(p.shape)
    #%pqq=pmarg(p,[1],wi);
    pqq = pmarg4(p, 1, 1, L, M)
    for i in range(2, L+1):
        il = i-q
        if il<1:
            il = 1
        
        
        pn = pmarg4(p, il, i, L, M)
        pd = pmarg4(p, il, (i-1), L, M)
        pqq = pqq*pn/(pd+EPS)
        
    return pqq
    