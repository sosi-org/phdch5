#Is used
import numpy as np
#import scipy
#import matcompat
#import matplotlib.pylab as plt
from consts import EPS
#from probrs_dith import probrs_dith
#todo: rename this function/file

def my_prs_dith1(spk, nt):

    # Local Variables: ps, spk_c, pra, bias, h0, ntr, M, L, spk, t, prs, DEBUG, ns, nt, prs1
    # Function calls: all, max, sum, eps, abs, reshape, my_prs_dith1, probrs_dith, size
    #%no method_id here
    #%spk values start with 0 (not 1)
    #% see Marcelo's    hrs.m
    #%q=hrs();
    DEBUG = True
    ns = spk.shape[4-1]
    L = spk.shape[2-1]
    h0 = 0.
    #%err=0;
    #M = 1.+matcompat.max(np.reshape(spk, 1., np.array([])))
    M = 1+max(spk.flatten())
    if M == 1:
        M = 2
    assert min(spk.flatten()) >= 0
    
    
    pra = np.array([])
    for t in np.arange(1., (ns)+1):
        #%over all stimulus conditions


        #%Direct estimation
        ntr = nt[int(t)-1]
        if not DEBUG:
            #%sohail
            #    assert(ntr>=2); %sohail
            pass

        spk_c = spk
        spk_c[(spk_c[:] > M-1.)] = M-1.
        spk_c[(spk_c[:]<0.)] = 0.
        #%trials=(squeeze(spk(1,:,:,t)))'; %trials set for current stimulus condition
        #  prs1=probrs_dith(spk_c,[1:ntr],t,M);  %it uses the same probr functions as Hr does!
        prs1 = probrs_dith(spk_c, np.array(np.hstack((np.arange(1., (ntr)+1)))), t, M)
        #%it uses the same probr functions as Hr does!
        #% hdt=-sum(prs.*log2(prs+eps))*nt(t);
        #assert(abs( sum(prs1(:)) - 1 ) < EPS*100);
        assert abs( sum(prs1.flatten()) - 1 ) < EPS*100

        #\'
        bias = 0.
        if t == 1.:
            pra = prs1.T
            #% 1x64 -> 64x1
        else:
            pra[:,int(t)-1] = prs1
        #assert pra.shape == (,ns)
        
        #% h0=h0+hdt;
        #return [prs, ps]

    #assert False
    #% h0=h0/sum(nt);
    #ps = matdiv(nt.flatten(1), np.sum(nt))
    ps = nt.flatten(1) /  float(np.sum(nt))
    #% for   NS x 1  (and not the transpose)
    #%up to 2e-14. why??
    #assert(abs( sum(ps) - 1 ) < 1e-13 ); %up to 2e-14. why??
    assert abs( sum(ps) - 1 ) < 1e-9 #) ; %up to 2e-14. why??

    prs = pra
    return (prs, ps)

