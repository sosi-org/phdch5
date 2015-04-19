
import numpy as np
#import scipy
#import matcompat
#import matplotlib.pylab as plt
from consts import EPS
from range_shuffle import range_shuffle
from probr import probr
from probqr import probqr
from range_frac import range_frac

def xiq(spk, nt, q, biastype):
    assert biastype<=1

    # Local Variables: h44, p43q, h41, biastype, h2, h0, p22, h4, r44, bias, r41, h42, r43, r42, p44, p43, r22, p41, out, p21q, h22, p41q, h21, p42, rg, r21, ntrg, nt, p42q, p22q, L, p21r, pqq, h43, hc0, g, ntr4, j, ntr2, p44q, q, p, _srange0, spk, p21, p21i
    # Function calls: range_shuffle, rand, log2, lagrange2, floor, sum, lagrange3, eps, xiq, range_frac, probqr, probr, size
    #%This function estimates xiq of a set of trials
    #%The result is given in bits
    #%The estimator implemented is chosen by biastype:
    #%Bias correction
    #%0: direct
    #%1= cuadratic extrap
    #%2= naive NOT IMPLEMENTED
    #%3= Panzeri NOT IMPLEMENTED
    #%4= Montemurro NOT IMPLEMENTED
    #%5= Nemenman NOT IMPLEMENTED
    #L = matcompat.size(spk, 2.)
    #%ntr=size(spk,3);
    #%_srange0=[1:ntr];

    L=spk.shape[1]
    _srange0 = range_shuffle(nt)
    #%M=1+max(reshape(spk,1,[]));
    p = probr(spk, nt, _srange0, 1)
    pqq = probqr(spk, nt, _srange0, q, 1)
    bias = 0.
    if biastype >= 1:
        bias = 1
    if biastype == 8:
        bias = 8 #correct
    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%Direct estimation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    hc0 = -np.sum((p*np.log2((pqq+EPS))))
    _switch_val=bias
    if _switch_val == 0:
        bias = 0.
        h0 = hc0
    elif _switch_val == 1:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        _srange0 = range_shuffle(nt)
        ntr2 = np.floor((np.sum(nt)/2.))
        ntr4 = np.floor((np.sum(nt)/4.))
        r21 = range_frac(_srange0, nt, 2., 1.)
        r22 = range_frac(_srange0, nt, 2., 2.)
        r41 = range_frac(_srange0, nt, 4., 1.)
        r42 = range_frac(_srange0, nt, 4., 2.)
        r43 = range_frac(_srange0, nt, 4., 3.)
        r44 = range_frac(_srange0, nt, 4., 4.)
        p21 = probr(spk, nt, r21, 2.)
        p22 = probr(spk, nt, r22, 2.)
        p41 = probr(spk, nt, r41, 4.)
        p42 = probr(spk, nt, r42, 4.)
        p43 = probr(spk, nt, r43, 4.)
        p44 = probr(spk, nt, r44, 4.)
        p21q = probqr(spk, nt, r21, q, 2.)
        p22q = probqr(spk, nt, r22, q, 2.)
        p41q = probqr(spk, nt, r41, q, 4.)
        p42q = probqr(spk, nt, r42, q, 4.)
        p43q = probqr(spk, nt, r43, q, 4.)
        p44q = probqr(spk, nt, r44, q, 4.)
        h21 = -np.sum((p21*np.log2((p21q+EPS))))
        h22 = -np.sum((p22*np.log2((p22q+EPS))))
        h41 = -np.sum((p41*np.log2((p41q+EPS))))
        h42 = -np.sum((p42*np.log2((p42q+EPS))))
        h43 = -np.sum((p43*np.log2((p43q+EPS))))
        h44 = -np.sum((p44*np.log2((p44q+EPS))))
        h4 = (h41+h42+h43+h44)/4.
        h2 = (h21+h22)/2.
        #%h0=(8*hc0-6*h2+h4)/3; %parabolic extrapolation
        #%h0=(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hd*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
        h0 = lagrange3(np.array(np.hstack((1./ntr4, 1./ntr2, 1./np.sum(nt)))), np.array(np.hstack((h4, h2, hc0))), 0.)
        #%hst=(4*hd-h21-h22)/2; %linear extrapolation
    elif _switch_val == 2.:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Naive correction
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        h0 = 0.
    elif _switch_val == 3.:
        #%%%%%%%%%%%%%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Panzeri
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        h0 = 0.
    elif _switch_val == 4.:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Montemurro
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        h0 = 0.
    elif _switch_val == 5.:
        #%first recover absolute freqs
        h0 = 0.
    elif _switch_val == 7.:
        _srange0 = range_shuffle(nt)
        g = 5.
        ntrg = nt[0]-1.
        h21 = 0.
        for j in np.arange(1., (g)+1):
            out = np.floor(np.dot(nt[0], np.random.rand(1., 1.)))+1.
            rg = _srange0
            rg[:,int(out)-1] = np.array([])
            p21r = probr(spk, (nt-1.), rg, 1.)
            p21i = probqr(spk, (nt-1.), rg, q, 1.)
            h21 = h21-np.sum((p21r*np.log2((p21i+EPS))))
            
        h21 = matdiv(h21, g)
        h0 = lagrange2(np.array(np.hstack((1./ntrg, 1./nt[0]))), np.array(np.hstack((h21, hc0))), 0.)
    elif _switch_val == 8.:
        _srange0 = range_shuffle(nt)
        ntr2 = np.floor((np.sum(nt)/2.))
        r21 = range_frac(_srange0, nt, 2., 1.)
        r22 = range_frac(_srange0, nt, 2., 2.)
        p21 = probr(spk, nt, r21, 2.)
        p22 = probr(spk, nt, r22, 2.)
        p21q = probqr(spk, nt, r21, q, 2.)
        p22q = probqr(spk, nt, r22, q, 2.)
        h21 = -np.sum((p21*np.log2((p21q+EPS))))
        h22 = -np.sum((p22*np.log2((p22q+EPS))))
        h2 = (h21+h22)/2.
        #%h0=(8*hc0-6*h2+h4)/3; %parabolic extrapolation
        #%h0=(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hd*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
        h0 = lagrange2(np.array(np.hstack((1./ntr2, 1./np.sum(nt)))), np.array(np.hstack((h2, hc0))), 0.)
    
    #%error estimation, Latham's
    #%N=ntr;
    #%err=sqrt((sum(p.*log2(p+eps).^2)-(hd*L)^2)/(L*N));
    return h0