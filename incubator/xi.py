
import numpy as np
#import scipy
#import matcompat
from consts import EPS
#import matplotlib.pylab as plt

from range_shuffle import range_shuffle
from probr import probr
from range_frac import range_frac
from probrind import probrind
from lagrange2 import lagrange2
from lagrange3 import lagrange3

def xi(spk, nt, biastype):
    #print nt
    # Local Variables: h44, j, h41, biastype, h2, h21, spk, h4, r44, bias, r41, h42, r43, r42, r22, r21, out, pr, p43i, p44i, prind, p42r, h22, rg, p42i, ns, nt, p44r, p22r, ntr, L, p22i, p41r, ntrg, g, ntr4, h, p43r, ntr2, xi0, p21i, _srange0, p21r, h43, p41i
    # Function calls: range_shuffle, rand, xi, log2, lagrange2, floor, sum, lagrange3, eps, probrind, range_frac, probr, size
    #%03/05/2005 quadratic extrapolation added
    ntr = spk.shape[2] #matcompat.size(spk, 3)
    ns = spk.shape[3] #matcompat.size(spk, 4)
    L = spk.shape[1] #matcompat.size(spk, 2)
    #%trials=(reshape(spkt,L,[]))'; %comprising all stimulus conditions
    _srange0 = range_shuffle(nt)

    #%M=1+max(reshape(spk,1,[]));
    pr = probr(spk, nt, _srange0, 1)


    prind = probrind(spk, nt, _srange0, 1)



    h = -np.sum((pr*np.log2((prind+EPS))))


    bias = 0.
    if biastype >= 1.:
        bias = 1.
        assert False
    
    
    if biastype == 8.:
        bias = 8.
        assert False
    
    
    _switch_val=bias
    if False: # switch 
        pass
    elif _switch_val == 0.:
        xi0 = h
    elif _switch_val == 1.:
        #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
        #%creates _srange0 matrix
        _srange0 = range_shuffle(nt)
        ntr2 = np.floor((np.sum(nt)/2.))
        ntr4 = np.floor((np.sum(nt)/4.))
        r21 = range_frac(_srange0, nt, 2., 1.)
        r22 = range_frac(_srange0, nt, 2., 2.)
        r41 = range_frac(_srange0, nt, 4., 1.)
        r42 = range_frac(_srange0, nt, 4., 2.)
        r43 = range_frac(_srange0, nt, 4., 3.)
        r44 = range_frac(_srange0, nt, 4., 4.)
        p21r = probr(spk, nt, r21, 2.)
        p22r = probr(spk, nt, r22, 2.)
        p41r = probr(spk, nt, r41, 4.)
        p42r = probr(spk, nt, r42, 4.)
        p43r = probr(spk, nt, r43, 4.)
        p44r = probr(spk, nt, r44, 4.)
        p21i = probrind(spk, nt, r21, 2.)
        p22i = probrind(spk, nt, r22, 2.)
        p41i = probrind(spk, nt, r41, 4.)
        p42i = probrind(spk, nt, r42, 4.)
        p43i = probrind(spk, nt, r43, 4.)
        p44i = probrind(spk, nt, r44, 4.)
        h21 = -np.sum((p21r*np.log2((p21i+EPS))))
        h22 = -np.sum((p22r*np.log2((p22i+EPS))))
        h41 = -np.sum((p41r*np.log2((p41i+EPS))))
        h42 = -np.sum((p42r*np.log2((p42i+EPS))))
        h43 = -np.sum((p43r*np.log2((p43i+EPS))))
        h44 = -np.sum((p44r*np.log2((p44i+EPS))))
        h4 = (h41+h42+h43+h44)/4.
        h2 = (h21+h22)/2.
        #%xi0=(8*h-6*h2+h4)/3; %parabolic extrapolation
        #%hst=(4*h-h21-h22)/2; %linear extrapolation
        xi0 = lagrange3(np.array(np.hstack((1./ntr4, 1./ntr2, 1./np.sum(nt)))), np.array(np.hstack((h4, h2, h))), 0.)
    elif _switch_val == 2.:
        xi0 = 0.
    elif _switch_val == 3.:
        xi0 = 0.
    elif _switch_val == 4.:
        xi0 = 0.
    elif _switch_val == 5.:
        xi0 = 0.
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
            p21i = probrind(spk, (nt-1.), rg, 1.)
            h21 = h21-np.sum((p21r*np.log2((p21i+EPS))))
            
        #h21 = matdiv(h21, g)
        h21 = h21 / float(g)
        xi0 = lagrange2(np.array(np.hstack((1./ntrg, 1./nt[0]))), np.array(np.hstack((h21, h))), 0.)
    elif _switch_val == 8.:
        #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
        #%creates _srange0 matrix
        _srange0 = range_shuffle(nt)
        ntr2 = np.floor((np.sum(nt)/2.))
        r21 = range_frac(_srange0, nt, 2., 1.)
        r22 = range_frac(_srange0, nt, 2., 2.)
        p21r = probr(spk, nt, r21, 2.)
        p22r = probr(spk, nt, r22, 2.)
        p21i = probrind(spk, nt, r21, 2.)
        p22i = probrind(spk, nt, r22, 2.)
        h21 = -np.sum((p21r*np.log2((p21i+EPS))))
        h22 = -np.sum((p22r*np.log2((p22i+EPS))))
        h2 = (h21+h22)/2.
        #%xi0=(8*h-6*h2+h4)/3; %parabolic extrapolation
        #%hst=(4*h-h21-h22)/2; %linear extrapolation
        xi0 = lagrange2(np.array(np.hstack((1./ntr2, 1./np.sum(nt)))), np.array(np.hstack((h2, h))), 0.)
    
    return xi0
    