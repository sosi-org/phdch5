
import numpy as np
#import scipy
#import matplotlib.pylab as plt
from consts import EPS


def dh_ind(spikes):
    assert len(spikes.shape)==2
    # Local Variables: Nmax, count, spikes, i, h, L, wi, n, p
    # Function calls: log2, max, sum, eps, zeros, dh_ind, histc, size
    L = spikes.shape[2-1] #matcompat.size(spikes, 2)
    h = 0.
    for i in range(1, L+1):
        n = spikes[:,int(i)-1]
        Nmax = max(n) #matcompat.max(n)
        p = np.zeros([1, (Nmax+1)])
        count = np.zeros([Nmax+1, 1],int)
        wi = 1+n
        #    count=histc(wi,[1:Nmax+1+eps]);
        #count = histc(wi, np.array(np.hstack((np.arange(1., (Nmax+1.+EPS)+1)))))
        maxrange = (Nmax+1.+0.001)+1
        #edges=np.array(np.hstack((np.arange(1., maxrange))))
        #edges=np.array(np.arange(1, maxrange))
        edges=np.arange(1, maxrange)+0.1
        #count = histc(wi, edges)
        count,e2 = np.histogram(wi.flatten(), edges)
        assert sum(count) == spikes.shape[0]
        #p = matdiv(count, np.sum(count))
        p = count / float(np.sum(count))
        h = h-np.sum((p*np.log2((p+EPS))))        
    return h




    #maxrange = 1+(BIG_EPS+math.pow(M,L))
    #edges = np.array(range(0,int(maxrange)))+0.1 #EPS does did work here!! #bin edges, including the rightmost edge,
    #count,e2 = np.histogram(wi.flatten(), edges) #
    #assert sum(count) == sum(new_nta)
