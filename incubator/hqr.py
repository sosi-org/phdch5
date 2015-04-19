
import numpy as np
import scipy
import matcompat
from consts import EPS
import matplotlib.pylab as plt

#not fully processed

from range_shuffle import range_shuffle
from probqr import probqr
from range_frac import range_frac
from lagrange3 import lagrange3
from lagrange2 import lagrange2

def hqr(spk, nt, q, biastype):

    assert biastype <= 1
    assert type(q) is int
    # Local Variables: h44, h41, biastype, h2, h0, p22, h4, r44, bias, r41, h42, r43, r42, p44, p43, r22, r21, h22, h21, p42, p41, ns, nt, L, n1, h43, hc2, hc3, hc0, hc1, hc4, hc5, q, p, _srange0, spk, p21, n2, n4
    # Function calls: range_shuffle, log2, lagrange2, floor, sum, lagrange3, eps, range_frac, probqr, hqr, size
    #%This function estimates the response entropy of a set of trials
    #%The result is given in bits
    #%The estimator implemented is chosen by biastype:
    #%Bias correction
    #%hc0: direct
    #%hc1= cuadratic extrap
    #%hc2= naive NOT IMPLEMENTED
    #%hc3= Panzeri NOT IMPLEMENTED
    #%hc4= Montemurro NOT IMPLEMENTED
    #%hc5= Nemenman NOT IMPLEMENTED
    hc0 = 0.
    hc1 = 0.
    hc2 = 0.
    hc3 = 0.
    hc4 = 0.
    hc5 = 0.
    #%ntr=size(spk,3);
    #%spkt=squeeze(spk(1,:,:,:)); 
    L = matcompat.size(spk, 2.)
    #%trials=(reshape(spkt,L,[]))'; %comprising all stimulus conditions
    #%ntr=size(spk,3);
    ns = matcompat.size(spk, 4.)
    _srange0 = range_shuffle(nt)
    p = probqr(spk, nt, _srange0, q, 1)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%Direct estimation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    hc0 = -np.sum((p*np.log2((p+EPS))))
    bias = 0.
    _switch_val=biastype
    #if False: # switch
    #    pass
    #elif _switch_val == 1.:
    #    bias = 1.
    #elif _switch_val == 8.:
    #    bias = 8.
    #
    #
    _switch_val=bias
    #if False: # switch
    #    pass
    if _switch_val == 0.:
        bias = 0.
        h0 = hc0
    elif _switch_val == 1.:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        _srange0 = range_shuffle(nt)
        r21 = range_frac(_srange0, nt, 2., 1.)
        r22 = range_frac(_srange0, nt, 2., 2.)
        r41 = range_frac(_srange0, nt, 4., 1.)
        r42 = range_frac(_srange0, nt, 4., 2.)
        r43 = range_frac(_srange0, nt, 4., 3.)
        r44 = range_frac(_srange0, nt, 4., 4.)
        p21 = probqr(spk, nt, r21, q, 2.)
        p22 = probqr(spk, nt, r22, q, 2.)
        p41 = probqr(spk, nt, r41, q, 4.)
        p42 = probqr(spk, nt, r42, q, 4.)
        p43 = probqr(spk, nt, r43, q, 4.)
        p44 = probqr(spk, nt, r44, q, 4.)
        h21 = -np.sum((p21*np.log2((p21+EPS))))
        h22 = -np.sum((p22*np.log2((p22+EPS))))
        h41 = -np.sum((p41*np.log2((p41+EPS))))
        h42 = -np.sum((p42*np.log2((p42+EPS))))
        h43 = -np.sum((p43*np.log2((p43+EPS))))
        h44 = -np.sum((p44*np.log2((p44+EPS))))
        h4 = (h41+h42+h43+h44)/4.
        h2 = (h21+h22)/2.
        n1 = np.sum(nt)
        n2 = np.sum(np.floor((nt/2.)))
        n4 = np.sum(np.floor((nt/4.)))
        #%h0=(8*hc0-6*h2+h4)/3; %parabolic extrapolation
        #h0 = lagrange3(np.array(np.hstack((1./n4, 1./n2, 1./n1))), np.array(np.hstack((h4, h2, hc0))), 0.)
        h0 = lagrange3(np.array([1./n4, 1./n2, 1./n1]), np.array([h4, h2, hc0]), 0.)
        #%h0=(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hd*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
        #%hst=(4*hd-h21-h22)/2; %linear extrapolation
    elif _switch_val == 2.:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Naive correction
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        h0 = 0.
    elif _switch_val == 3.:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Panzeri
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        h0 = 0.
    elif _switch_val == 4.:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Montemurro
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        h0 = 0.
    elif _switch_val == 5.:
        h0 = 0.
        #%first recover absolute freqs
        #%THIS IS A CHEAT
        #%n=round(ntr*ns*p);
        #%[h0]=saddleentr2(n);;
    elif _switch_val == 8.:
        _srange0 = range_shuffle(nt)
        r21 = range_frac(_srange0, nt, 2., 1.)
        r22 = range_frac(_srange0, nt, 2., 2.)
        p21 = probqr(spk, nt, r21, q, 2.)
        p22 = probqr(spk, nt, r22, q, 2.)
        h21 = -np.sum((p21*np.log2((p21+EPS))))
        h22 = -np.sum((p22*np.log2((p22+EPS))))
        h2 = (h21+h22)/2.
        #%h0=(8*hc0-6*h2+h4)/3; %parabolic extrapolation
        n1 = np.sum(nt)
        n2 = np.sum(np.floor((nt/2.)))
        #h0 = lagrange2(np.array(np.hstack((1./n2, 1./n1))), np.array(np.hstack((h2, hc0))), 0.)
        h0 = lagrange2(np.array([1./n2, 1./n1]), np.array([h2, hc0] ), 0.)

    #%error estimation, Latham's
    #%N=ntr;
    #%err=sqrt((sum(p.*log2(p+eps).^2)-(hd*L)^2)/(L*N));
    return h0
    