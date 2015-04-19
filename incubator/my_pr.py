
import numpy as np
#import scipy
#import matcompat
#import matplotlib.pylab as plt

def my_pr(prsa, ps):

    # Local Variables: pr, ps, prsa
    # Function calls: my_pr, length, size
    #if matcompat.size(ps, 1.) == 1.:
    #    ps = ps.conj().T
    #assert ps.shape==(,)
    
    #assert ps.shape == (prsa.shape[1],1) #not tested
    assert ps.shape == (prsa.shape[1],) #one dimensional
    #%rdim = size(prsa,1);
    #%sdim = size(prsa,2);
    pr = np.dot(prsa, ps)
    #%Matrix? !
    #% why hr.m uses: range=range_shuffle(nt);
    return pr
