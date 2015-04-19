import numpy as np
import scipy
#import matcompat
from consts import EPS


def q_resamp(spk, nti, nto, q):

    # Local Variables: rho, iu, spk, id, trials, pd, memory, ns, pn, spks, L, nti, nto, idx2, idx1, a, i, ml, k, m, q, p, t, z
    # Function calls: rand, q_resamp, max, sum, eps, pqmargs, zeros, squeeze, size
    #%it resamples the given experiment keeping
    #%statistical structure up to order q
    #%nti: vector with the number of trials per stimulus in the input experiment
    #%nto: vector with the number of trials per stimulus in the output resampled experiment
    #%ntr=size(spk,3);
    ns = spk.shape[4-1] #matcompat.size(spk, 4.)
    L = spk.shape[2-1] #matcompat.size(spk, 2.)
    m = max(nto)
    spks = np.zeros([1, L, m, ns], int)
    for t in range(1, (ns)+1):
        #%over all stimulus conditions

        #%over all stimulus conditions
        trials = np.squeeze(spk[0,:,0:nti[int(t)-1],int(t)-1]).conj().T
        #%' 
        #%trials set for current stimulus condition
        if L == 1:
            trials = trials.conj().T
            #%'
        
        
        [pn, pd] = pqmargs(trials, L, q)
        for k in np.arange(1, (nto[int(t)-1])+1):
            rho = np.zeros([1, L], int)
            #%now generate the samples
            #%probability of getting a 1 in bin 1 is     1-pn(1,1)
            z = np.random.rand(1., 1.)
            rho[0] = z<1.-pn[0,0]
            for i in np.arange(2., (L)+1):
                iu = i-1
                id = i-q
                if id<1:
                    id = 1.
                
                
                memory = rho[int(id)-1:iu]
                ml = iu-id+1.
                #idx1 = np.sum((matixpower(2., np.array(np.hstack((np.arange(0., (ml-1.)+1)))))*memory))+1.
                idx1 = np.sum(((2 ** np.arange(0, ml-1+1))*memory))+1
                idx2 = idx1+2**ml
                #%this is the index to the probability of getting a 1 in bin i given the memory
                p = matdiv(pn[int(i)-1,int(idx2)-1], pd[int(i)-1,int(idx1)-1]+EPS)
                z = np.random.rand(1., 1.)
                rho[int(i)-1] = z<p
                
            spks[0,:,int(k)-1,int(t)-1] = rho
            
    return spks
    