#todo
import numpy as np
#import scipy
#import matcompat

import matplotlib.pylab as plt
from consts import EPS
from lagrange2 import lagrange2
from dh_ind import dh_ind
from consts import EPS
from lagrange3 import lagrange3
from range_shuffle import range_shuffle

def matdev(a,b):
    return a/b

def randperm(n):
    return np.random.permutation(range(n))+1

def hrsind(spk, nt, biastype):

    # Local Variables: h44, h41, biastype, h2, h0, prsind, h4, r44, spk, r41, h42, r43, r42, r22, r21, bias1, spk22, ntrt, trials, h00, h21, f21, h22, bias0, idx, spk21, spk41, hnemt, spk43, ns, nt, spk44, spk42, dh, bias2, ntr, L, R, hc2, hc3, hc0, hc1, betac, hc4, err, hc5, ntr4, f22, ntr2, n, i, _range, bias, t, h43
    # Function calls: range_shuffle, lagrange3, simps_quad, f, log2, log, lagrange2, floor, sum, randperm, eps, hrsind, zeros, saddleentr3, bayescount, dh_ind, squeeze, round, size
    #%It will estimate the entropy of a binary chain giving the result in bits per bin
    #%trials=squeeze(spk(1,:,:))';
    #%Bias correction
    #%hc0: direct
    #%hc1= cuadratic extrap
    #%hc2= naive
    #%hc3= Panzeri
    #%hc4= Montemurro
    #%hc5= Nemenman
    if biastype > 0:
        bias = 1
    else:
        bias = 0
        
    
    hc0 = 0.
    hc1 = 0.
    hc2 = 0.
    hc3 = 0.
    hc4 = 0.
    hc5 = 0.
    ntrt = np.sum(nt)
    ns = spk.shape[3] #ns = matcompat.size(spk, 4.)
    L = spk.shape[1] #L = matcompat.size(spk, 2.)
    h0 = 0.
    err = 0.
    if False:
        _range = range_shuffle(nt) #Why not used???

    for t in range(1, (ns)+1):
        #%over all stimulus conditions




        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Direct estimation
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        
        #trials1 = spk[:,0:L,0:nt[t-1],t-1:t] #  1x1x16x1
        #trials1 = spk[0,0:L,0:nt[t-1],t-1:t] #  1x16x1
        #trials1 = spk[0,0:L,0:nt[t-1],t-1] #  1x16
        #print trials1.shape
        #print trials1.T.shape

        #trials = np.squeeze(spk[0,:,0:nt[t-1],t-1]).T
        trials = spk[0,0:L,0:nt[t-1],t-1].T
        assert trials.shape == (nt[t-1],L)
        #    #%trials set for current stimulus condition
        #    if L == 1:
        #        #trials = trials.conj().T
        #        trials = trials.reshape( (nt[t-1],L) )
        #        #todo: a better indexing/slicing to avoid the need for this.
        #    
        #    assert trials.shape == (nt[t-1],L)


        dh = dh_ind(trials)
        _switch_val=bias
        if False: # switch
            pass
        elif _switch_val == 0.:
            bias = 0.
            h0 = h0+np.dot(dh, nt[int(t)-1])
        elif _switch_val == 1.:
            #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
            ntr = nt[int(t)-1]
            idx = randperm(ntr)
            ntr2 = np.floor((ntr/2.))
            ntr4 = np.floor((ntr/4.))
            r21 = idx[0:ntr2]
            r22 = idx[int(ntr2+1.)-1:2.*ntr2]
            r41 = idx[0:ntr4]
            r42 = idx[int(ntr4+1.)-1:2.*ntr4]
            r43 = idx[int(2.*ntr4+1.)-1:3.*ntr4]
            r44 = idx[int(3.*ntr4+1.)-1:4.*ntr4]
            ntr2 = np.floor((nt[int(t)-1]/2.))
            ntr4 = np.floor((nt[int(t)-1]/4.))
            spk21 = trials[int(r21)-1,:]
            spk22 = trials[int(r22)-1,:]
            spk41 = trials[int(r41)-1,:]
            spk42 = trials[int(r42)-1,:]
            spk43 = trials[int(r43)-1,:]
            spk44 = trials[int(r44)-1,:]
            h21 = dh_ind(spk21)
            #%-sum(f21.*log2(f21+eps)+(1-f21).*log2(1-f21+eps));
            h22 = dh_ind(spk22)
            #%-sum(f22.*log2(f22+eps)+(1-f22).*log2(1-f22+eps));
            h41 = dh_ind(spk41)
            #%-sum(f41.*log2(f41+eps)+(1-f41).*log2(1-f41+eps));
            h42 = dh_ind(spk42)
            #%-sum(f42.*log2(f42+eps)+(1-f42).*log2(1-f42+eps));
            h43 = dh_ind(spk43)
            #%-sum(f43.*log2(f43+eps)+(1-f43).*log2(1-f43+eps));
            h44 = dh_ind(spk44)
            #%-sum(f44.*log2(f44+eps)+(1-f44).*log2(1-f44+eps));
            h4 = np.dot(nt[int(t)-1], h41+h42+h43+h44)/4.
            h2 = np.dot(nt[int(t)-1], h21+h22)/2.
            h0 = h0+lagrange3(np.array(np.hstack((1./ntr4, 1./ntr2, 1./ntr))), np.array(np.hstack((h4, h2, np.dot(dh, ntr)))), 0.)
            #%h0=h0+(8*dh-6*h2+h4)/3; %parabolic extrapolation
        elif _switch_val == 2.:
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Naive
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            bias = 0.
            prsind = np.zeros( (1,2))
            for i in np.arange(1, (L)+1):
                prsind[0] = plt.f(i)
                #%this is the bin probability of '1'
                prsind[1] = 1.-plt.f(i)
                bias = bias+matdiv(np.sum((prsind > EPS))-1., np.dot(2.*nt[int(t)-1], np.log(2.)))

            h0 = h0+dh+np.dot(bias, nt[int(t)-1])
        elif _switch_val == 3.:
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Panzeri
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            bias = 0.
            prsind = np.zeros( (1,2))
            for i in np.arange(1., (L)+1):
                prsind[0] = plt.f(i)
                #%this is the bin probability of '1'
                prsind[1] = 1.-plt.f(i)
                raise Exception("Not impplemented")
                R = bayescount(nt[int(t)-1], prsind)
                bias = bias+matdiv(R-1., np.dot(2.*nt[int(t)-1], np.log(2.)))

            h0 = h0+dh+np.dot(bias, nt[int(t)-1])
        elif _switch_val == 4.:
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Montemurro
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            raise Exception("Not implemented")
            ntr = nt[int(t)-1]
            idx = randperm(ntr)
            #%it performs a random permutation of the indeces to trials
            ntr2 = np.floor((ntr/2.))
            r21 = idx[0:ntr2]
            r22 = idx[int(ntr2+1.)-1:2.*ntr2]
            spk21 = trials[int(r21)-1,:]
            spk22 = trials[int(r22)-1,:]
            f21 = matdiv(np.sum(spk21), ntr2)
            f22 = matdiv(np.sum(spk22), ntr2)
            h21 = -np.sum((f21*np.log2((f21+EPS))+(1.-f21)*np.log2((1.-f21+EPS))))
            h22 = -np.sum((f22*np.log2((f22+EPS))+(1.-f22)*np.log2((1.-f22+EPS))))
            bias0 = 0.
            bias1 = 0.
            bias2 = 0.
            prsind = np.zeros((1,2))
            for i in np.arange(1., (L)+1):
                prsind[0] = plt.f(i)
                #%this is the bin probability of '1'
                prsind[1] = 1.-plt.f(i)
                bias0 = bias0+matdiv(np.sum((prsind > EPS))-1., np.dot(2.*ntr, np.log(2.)))
                prsind[0] = f21[int(i)-1]
                #%this is the bin probability of '1'
                prsind[1] = 1.-f21[int(i)-1]
                bias1 = bias1+matdiv(np.sum((prsind > EPS))-1., np.dot(2.*ntr2, np.log(2.)))
                prsind[0] = f22[int(i)-1]
                #%this is the bin probability of '1'
                prsind[1] = 1.-f22[int(i)-1]
                bias2 = bias2+matdiv(np.sum((prsind > EPS))-1., np.dot(2.*ntr2, np.log(2.)))

            h21 = h21+bias1
            h22 = h22+bias2
            h2 = (h21+h22)/2.
            h00 = dh+bias0
            #%h0=h0+(4*h00-h21-h22)/2;
            h0 = h0+lagrange2(np.array(np.hstack((1./ntr2, 1./ntr))), np.array(np.hstack((h2, h00))), 0.)
        elif _switch_val == 5.:
            #%Nemmenman
            #%first recover absolute freqs
            dh = 0.
            n = np.zeros((1., 2.))
            for i in np.arange(1., (L)+1):
                n[0] = np.round(np.dot(nt[int(t)-1], plt.f(i)))
                n[1] = np.round(np.dot(nt[int(t)-1], 1.-plt.f(i)))
                [hnemt] = saddleentr3(n)
                dh = dh+hnemt

            h0 = h0+np.dot(dh, nt[int(t)-1])
        elif _switch_val == 6.:
            #%Nemmenman
            #%first recover absolute freqs
            dh = 0.
            n = np.zeros((1., 2.))
            for i in np.arange(1., (L)+1):
                n[0] = np.round(np.dot(nt[int(t)-1], plt.f(i)))
                n[1] = np.round(np.dot(nt[int(t)-1], 1.-plt.f(i)))
                [hnemt] = simps_quad(n, betac[0,:])
                dh = dh+hnemt

            h0 = h0+np.dot(dh, nt[int(t)-1])
        elif _switch_val == 8.:
            ntr = nt[int(t)-1]
            idx = randperm(ntr)
            ntr2 = np.floor((ntr/2.))
            r21 = idx[0:ntr2]
            r22 = idx[int(ntr2+1.)-1:2.*ntr2]
            ntr2 = np.floor((nt[int(t)-1]/2.))
            spk21 = trials[int(r21)-1,:]
            spk22 = trials[int(r22)-1,:]
            f21 = matdiv(np.sum(spk21), ntr2)
            f22 = matdiv(np.sum(spk22), ntr2)
            h21 = -np.sum((f21*np.log2((f21+EPS))+(1.-f21)*np.log2((1.-f21+EPS))))
            h22 = -np.sum((f22*np.log2((f22+EPS))+(1.-f22)*np.log2((1.-f22+EPS))))
            h2 = np.dot(nt[int(t)-1], h21+h22)/2.
            h0 = h0+lagrange2(np.array(np.hstack((1./ntr2, 1./ntr))), np.array(np.hstack((h2, dh))), 0.)
            #%h0=h0+(8*dh-6*h2+h4)/3; %parabolic extrapolation

        #%swithc
        #return h0




    h0 = h0 / ntrt
    return h0