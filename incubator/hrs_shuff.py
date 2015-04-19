
import numpy as np
import scipy
import matcompat

from hrs import hrs
import matplotlib.pylab as plt

def randperm(n):
    return np.random.permutation(range(n))+1

def hrs_shuff(spk, nt, biastype):

    # Local Variables: hc2, hc3, hc0, hc1, betac, err, hc4, biastype, ntr, h0, hc5, L, s, spk, bini, ns, nt, k, spks
    # Function calls: hrs_suff, randperm, hrs, size
    #%It will estimate the entropy of a binary chain giving the result in bits per bin
    #%trials=squeeze(spk(1,:,:))';
    #%Bias correction
    #%hc0: direct
    #%hc1= cuadratic extrap
    #%hc2= naive
    #%hc3= Panzeri
    #%hc4= Montemurro
    #%hc5= Nemenman
    hc0 = 0.
    hc1 = 0.
    hc2 = 0.
    hc3 = 0.
    hc4 = 0.
    hc5 = 0.
    ntr = matcompat.size(spk, 3.)
    ns = matcompat.size(spk, 4.)
    L = matcompat.size(spk, 2.)
    h0 = 0.
    err = 0.

    spks = np.zeros([1,L,ntr,ns],int)

    for s in range(1, int(ns)+1):
        #%over all stimulus conditions
        for k in range(1,L+1):
           #bini=randperm(nt(s));
           #spks(1,k,1:nt(s),s)=spk(1,k,bini,s);
           bini = randperm(nt[s-1]);
           spks[1-1,k-1,0:nt[s-1],s-1] = spk[1-1,k-1,bini-1,s-1];

    
    #assert False
    h0 = hrs(spks, nt, biastype)
    return h0
    