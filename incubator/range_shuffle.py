
import numpy as np
import scipy
import matcompat
import matplotlib.pylab as plt
from  hx_types import *

def randperm(n):
    #print type(n),
    #print n
    return np.random.permutation(range(n))+1

def range_shuffle(nta):
    """ Creates a 2D array of indices based on the given nta """

    # Function calls: range_shuffle, max, length, randperm, zeros
    #%this function shuffles the index matrix for all stimulus conditions
    #print type(nt[0])
    assert type(nta[0]) in [INT_TYPE], "%r"%(type(nta[0])) #not int, probr cannot tolerate 'list's
    m = max(nta)
    ns = len(nta)
    idx = np.zeros((ns, m),int)
    for s in range(1, (ns)+1):
        idx[int(s)-1,0:nta[int(s)-1]] = randperm(nta[int(s)-1])
        
    return idx