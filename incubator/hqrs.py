
import numpy as np
import scipy
#import matcompat
from hmarg import hmarg
from range_shuffle import range_shuffle
from pqmargs import pqmargs
from lagrange3 import lagrange3
from hx_types import is_any_int_type
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

from consts import EPS
#EPS = finfo(float).eps
#EPS = 0.0000000000001

def randperm(n):
    return np.random.permutation(range(n))+1

def hqrs(spk, nt, q, biastype):
    assert biastype < 2
    # Local Variables: h41, biastype, h2, h0, iu, h4, Rd, r44, spk, r41, h42, r43, r42, id, Rn, n, r22, r21, out, bias0, p44n, le, p43d, p44d, trials, h21, h22, p42n, rg, g, pd, nn, hnemt, p42d, ns, pn, spkt, ntrg, p43n, K, ntr, L, bias22, bias, R, p22n, p22d, hg, p21d, hdt, betac, err, idx, ntr4, nt, j, bias21, ntr2, p21n, q, i, _srange, h44, t, p41d, h43, p41n, il
    # Function calls: range_shuffle, lagrange3, simps_quad, log, lagrange2, floor, reshape, sum, randperm, hqrs, eps, pqmargs, bayescount, saddleentr3, hmarg, squeeze, rand, round, size
    #%It will estimate the entropy of a binary chain giving the result in bits per bin
    #%Needs cleaning hs=0, hst has been removed 7/6/05
    #%anotated by Sohail
    #%Bias correction
    #%hc0: direct
    #%hc1= cuadratic extrap
    #%hc2= naive
    #%hc3= Panzeri
    #%hc4= Montemurro
    #%hc5= Nemenman
    #%ntr=size(spk,3);
    ns = spk.shape[3] #ns = matcompat.size(spk, 4.)
    L=spk.shape[1] #L = matcompat.size(spk, 2.)
    h0 = 0.
    err = 0.
    #%Selects indeces ranges for sample size corrections
    _srange = range_shuffle(nt)
    #for t in np.arange(1, (ns)+1):
    for t in range(1, ns+1):
        #%compute P(r|s)

        #%compute P(r|s)
        #spkt=squeeze(spk(1,:,[1:nt(t)],t));  % L x ntrials
        #trials=(reshape(spkt,L,[]))'; % ntrials x L
        #[pn,pd]=pqmargs(trials,L,q);
        #hdt=hmarg(pn,pd,L)*nt(t);



        #trials=(reshape(spkt,L,[]))'; % ntrials x L
        #[pn,pd]=pqmargs(trials,L,q);
        #hdt=hmarg(pn,pd,L)*nt(t);
        #print spk[0,0,0,0], "*****"
        assert is_any_int_type(spk[0,0,0,0])

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%compute P(r|s)
        #spkt=squeeze(spk(1,:,[1:nt(t)],t));  % L x ntrials
        #spkt = np.squeeze(spk[0,:,np.array(np.hstack((0:nt[int(t)-1]))),int(t)-1])
        #spkt = np.squeeze(spk[0,:,0:nt[t-1],int(t)-1])
        spkt = spk[0,:,0:nt[t-1],t-1]
        #print spkt.shape
        assert spkt.shape==(L,nt[t-1]) #2x16
        #% L x ntrials
        #trials = np.reshape(spkt, L, np.array([])).conj().T
        #trials = np.reshape(spkt, L, np.array([])).conj().T
        trials = spkt.T
        assert trials.shape==(nt[t-1],L)
        #% ntrials x L
        #print L,q
        assert is_any_int_type(trials[0])
        #assert type(trials[0]) in [int,np.int64,np.int16]
        [pn, pd] = pqmargs(trials, L, q)
        #hdt = np.dot(hmarg(pn, pd, L), nt[int(t)-1])
        hdt = hmarg(pn, pd, L) * nt[int(t)-1]
        #print type(hdt) #scalar
        #assert False
        _switch_val=biastype
        #FIXME: below is not processed
        #if False: # switch
        #    pass
        if _switch_val == 0.:
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Direct estimation
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            h0 = h0+hdt
        elif _switch_val == 1.:
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Quadratic extrapolation
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
            #%divide in ranges
            ntr = nt[int(t)-1]
            idx = randperm(ntr)
            ntr2 = np.floor((ntr/2))
            ntr4 = np.floor((ntr/4))
            r21 = idx[0:ntr2]
            r22 = idx[int(ntr2+1.)-1:2.*ntr2]
            r41 = idx[0:ntr4]
            r42 = idx[int(ntr4+1.)-1:2.*ntr4]
            r43 = idx[int(2.*ntr4+1.)-1:3.*ntr4]
            r44 = idx[int(3.*ntr4+1.)-1:4.*ntr4]
            ntr2 = np.floor((nt[int(t)-1]/2))
            ntr4 = np.floor((nt[int(t)-1]/4))
            #[p21n,p21d]=pqmargs(trials(r21,:),L,q);
            [p21n, p21d] = pqmargs(trials[int(r21)-1,:], L, q)
            [p22n, p22d] = pqmargs(trials[int(r22)-1,:], L, q)
            [p41n, p41d] = pqmargs(trials[int(r41)-1,:], L, q)
            [p42n, p42d] = pqmargs(trials[int(r42)-1,:], L, q)
            [p43n, p43d] = pqmargs(trials[int(r43)-1,:], L, q)
            [p44n, p44d] = pqmargs(trials[int(r44)-1,:], L, q)
            h21 = hmarg(p21n, p21d, L)
            h22 = hmarg(p22n, p22d, L)
            h41 = hmarg(p41n, p41d, L)
            h42 = hmarg(p42n, p42d, L)
            h43 = hmarg(p43n, p43d, L)
            h44 = hmarg(p44n, p44d, L)
            h4 = np.dot(nt[int(t)-1], h41+h42+h43+h44)/4.
            h2 = np.dot(nt[int(t)-1], h21+h22)/2.
            #%h0=h0+(8*hdt-6*h2+h4)/3;
            h0 = h0+lagrange3(np.array(np.hstack((1./ntr4, 1./ntr2, 1./nt[int(t)-1]))), np.array(np.hstack((h4, h2, hdt))), 0.)
            #%hst=hst+(4*hdt-h21-h22)/2;
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        elif _switch_val == 2.:
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Naive
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            R = np.sum((pn[0,:] > EPS))
            bias = matdiv(R-1., np.dot(2.*nt[int(t)-1], np.log(2.)))
            for i in np.arange(2., (L)+1):
                il = i-q
                if il<1.:
                    il = 1.


                Rn = np.sum((pn[int(i)-1,:] > EPS))
                Rd = np.sum((pd[int(i)-1,:] > EPS))
                bias = bias+matdiv(Rn-Rd, np.dot(2.*nt[int(t)-1], np.log(2.)))

            h0 = h0+hdt+np.dot(nt[int(t)-1], bias)
        elif _switch_val == 3.:
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Panzeri
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            R = bayescount(nt[int(t)-1], pn[0,0:2.])
            bias = R-1.
            for i in np.arange(2., (L)+1):
                il = i-q
                if il<1.:
                    il = 1.


                nn = 2.**(i-il+1.)
                Rn = bayescount(nt[int(t)-1], pn[int(i)-1,0:nn])
                Rd = bayescount(nt[int(t)-1], pd[int(i)-1,0:nn/2.])
                bias = bias+Rn-Rd

            h0 = h0+hdt+matdiv(np.dot(nt[int(t)-1], bias), np.dot(2.*nt[int(t)-1], np.log(2.)))
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Naive
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        elif _switch_val == 4.:
            [p21n, p21d] = pqmargs(trials[int(r21)-1,:], L, q)
            [p22n, p22d] = pqmargs(trials[int(r22)-1,:], L, q)
            h21 = hmarg(p21n, p21d, L)
            h22 = hmarg(p22n, p22d, L)
            bias0 = np.sum((pn[0,:] > EPS))-1.
            bias21 = np.sum((p21n[0,:] > EPS))-1.
            bias22 = np.sum((p22n[0,:] > EPS))-1.
            for i in np.arange(2., (L)+1):
                bias0 = bias0+np.sum((pn[int(i)-1,:] > EPS))-np.sum((pd[int(i)-1,:] > EPS))
                bias21 = bias21+np.sum((p21n[int(i)-1,:] > EPS))-np.sum((p21d[int(i)-1,:] > EPS))
                bias22 = bias22+np.sum((p22n[int(i)-1,:] > EPS))-np.sum((p22d[int(i)-1,:] > EPS))

            bias0 = matdiv(bias0, np.dot(2.*ntr, np.log(2.)))
            bias21 = matdiv(bias21, np.dot(ntr, np.log(2.)))
            bias22 = matdiv(bias22, np.dot(ntr, np.log(2.)))
            hdt = hdt+bias0
            h21 = h21+bias21
            h22 = h22+bias22
            h2 = (h21+h22)/2.
            #%h0=h0+(4*hdt-h21-h22)/2;
            h0 = h0+lagrange2(np.array(np.hstack((1./ntr2, 1./ntr))), np.array(np.hstack((h2, hdt))), 0.)
        elif _switch_val == 5.:
            #%Nemmenman
            #%first recover absolute freqs
            le = 1.
            n = np.round(np.dot(nt[int(t)-1], pn[0,:]))
            K = 2.**le
            n = n[0:K]
            [hnemt] = saddleentr3(n)
            h0 = h0+np.dot(hnemt, nt[int(t)-1])
            for i in np.arange(2., (L)+1):
                id = i-q
                iu = i
                if id<1.:
                    id = 1.


                le = iu-id+1.
                K = 2.**le
                n = np.round(np.dot(nt[int(t)-1], pn[int(i)-1,:]))
                n = n[0:K]
                [hnemt] = saddleentr3(n)
                h0 = h0+np.dot(hnemt, nt[int(t)-1])
                n = np.round(np.dot(nt[int(t)-1], pd[int(i)-1,0:K/2.]))
                [hnemt] = saddleentr3(n)
                h0 = h0-np.dot(hnemt, nt[int(t)-1])

        elif _switch_val == 6.:
            #%Nemmenman
            #%first recover absolute freqs
            #%first recover absolute freqs
            le = 1.
            n = np.round(np.dot(nt[int(t)-1], pn[0,:]))
            K = 2.**le
            n = n[0:K]
            [hnemt] = simps_quad(n, betac[int(le)-1,:])
            h0 = h0+np.dot(hnemt, nt[int(t)-1])
            for i in np.arange(2., (L)+1):
                id = i-q
                iu = i
                if id<1.:
                    id = 1.


                le = iu-id+1.
                K = 2.**le
                n = np.round(np.dot(nt[int(t)-1], pn[int(i)-1,:]))
                n = n[0:K]
                [hnemt] = simps_quad(n, betac[int(le)-1,:])
                h0 = h0+np.dot(hnemt, nt[int(t)-1])
                if q > 0.:
                    n = np.round(np.dot(nt[int(t)-1], pd[int(i)-1,0:K/2.]))
                    [hnemt] = simps_quad(n, betac[int((le-1.))-1,:])
                    h0 = h0-np.dot(hnemt, nt[int(t)-1])



        elif _switch_val == 7.:
            _srange = range_shuffle(nt)
            g = 5.
            ntrg = nt[int(t)-1]-1.
            hg = 0.
            for j in np.arange(1., (g)+1):
                rg = _srange[int(t)-1,0:nt[int(t)-1]]
                out = np.floor(np.dot(nt[int(t)-1], np.random.rand(1., 1.)))+1.
                rg[int(out)-1] = np.array([])
                [p21n, p21d] = pqmargs(trials[int(rg)-1,:], L, q)
                hg = hg+hmarg(p21n, p21d, L)

            h2 = matdiv(np.dot(nt[int(t)-1], hg), g)
            #%h0=h0+(8*hdt-6*h2+h4)/3;
            h0 = h0+lagrange2(np.array(np.hstack((1./ntrg, 1./nt[int(t)-1]))), np.array(np.hstack((h2, hdt))), 0.)
        elif _switch_val == 8.:
            ntr = nt[int(t)-1]
            idx = randperm(ntr)
            ntr2 = np.floor((ntr/2.))
            r21 = idx[0:ntr2]
            r22 = idx[int(ntr2+1.)-1:2.*ntr2]
            [p21n, p21d] = pqmargs(trials[int(r21)-1,:], L, q)
            [p22n, p22d] = pqmargs(trials[int(r22)-1,:], L, q)
            h21 = hmarg(p21n, p21d, L)
            h22 = hmarg(p22n, p22d, L)
            h2 = np.dot(nt[int(t)-1], h21+h22)/2.
            #%h0=h0+(8*hdt-6*h2+h4)/3;
            h0 = h0+lagrange2(np.array(np.hstack((1./ntr2, 1./nt[int(t)-1]))), np.array(np.hstack((h2, hdt))), 0.)

        
    h0 = h0 / np.sum(nt)
    return h0
