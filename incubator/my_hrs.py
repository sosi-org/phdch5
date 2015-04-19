from consts import EPS
import numpy as np
#import scipy
#import matcompat
#import matplotlib.pylab as plt

def my_hrs(prs, ps, methid):

    # Local Variables: ps, hrsa, methid, hrs, prs, NR, NS
    # Function calls: my_hrs, log2, sum, eps, length, size
    #NS = matcompat.size(prs, 2.)
    #NR = matcompat.size(prs, 1.)
    #if matcompat.size(ps, 1.) == 1.:
    #    ps = ps.conj().T
    NS = prs.shape[1] #matcompat.size(prs, 2.)
    NR = prs.shape[0] #matcompat.size(prs, 1.)
    if ps.shape[0]==1: #matcompat.size(ps, 1.) == 1.:
        ps = ps.conj().T
    assert ps.shape == (NS,)
    assert prs.shape[1]==ps.shape[0]

    
    #%ps : NS x 1
    #%NS, M, L, NR, ...
    #%ns=size(spk,4);
    #%ns=length(ps);
    #%L=size(spk,2);
    #%psa = repmat(ps,[1,NR]);
    #%prs : NR x NS
    hrsa = -np.sum((prs*np.log2((prs+EPS))), 1.)
    #% 1 x NS
    hrs = np.dot(hrsa, ps)
    #% Matrix multiplication!
    return hrs