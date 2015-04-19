"""
biastype <= 1 tested implemented
"""
import numpy as np
import scipy
import matcompat

#import matplotlib.pylab as plt
from lagrange2 import lagrange2

from  hx_types import *


from range_shuffle import range_shuffle
from probr import probr
from consts import EPS
#lagrange3

from range_frac import *
from lagrange3 import *



def hr(spk, nta, biastype):
    #nta was nt
    assert biastype <= 1

    # Local Variables: h44, h41, biastype, h2, h0, p22, h4, r44, spk, r41, h42, r43, r42, p44, p43, r22, r21, pp, h22, h00, h21, p42, bias0, bias1, bias2, p41, ns, nt, K, ntr, M, L, R, n1, h43, hc2, hc3, hc0, hc1, betac, idx, hc4, hc5, t2, ntr2, n, p, _range, bias, p21, n2, n4
    # Function calls: range_shuffle, simps_quad, randperm, log2, log, lagrange2, floor, hr, max, sum, lagrange3, eps, saddleentr3, range_frac, reshape, bayescount, probr, round, size
    #%This function estimates the response entropy of a set of trials
    #%The result is given in bits
    #%The estimator implemented is chosen by biastype:
    hc0 = 0.
    hc1 = 0.
    hc2 = 0.
    hc3 = 0.
    hc4 = 0.
    hc5 = 0.
    #L = matcompat.size(spk, 2.)
    L = spk.shape[2-1]
    #M = 1.+matcompat.max(np.reshape(spk, 1., np.array([])))
    M = 1+max(spk.flatten())
    if M == 1:
        M = 2
    
    #print type(nt)
    #assert type(nt) is list
    #assert type(nt) is np.array

    ntr = np.sum(nta)
    #%total number of trials
    ns = matcompat.size(spk, 4)
    assert type(ns) is int
    
    _range = range_shuffle(nta)
    p = probr(spk, nta, _range, 1)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%Direct estimation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    hc0 = -np.sum((p*np.log2((p+EPS))))
    _switch_val=biastype
    if False: # switch 
        pass
    elif _switch_val == 0:
        bias = 0.
        h0 = hc0
    elif _switch_val == 1:

        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #    %range=range_shuffle(nt);
        #    r21=range_frac(range,nt,2,1);
        #    r22=range_frac(range,nt,2,2);
        #    r41=range_frac(range,nt,4,1);


        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%_range=range_shuffle(nt);
        r21 = range_frac(_range, nta, 2, 1)
        r22 = range_frac(_range, nta, 2, 2)
        r41 = range_frac(_range, nta, 4, 1)
        r42 = range_frac(_range, nta, 4, 2)
        r43 = range_frac(_range, nta, 4, 3)
        r44 = range_frac(_range, nta, 4, 4)

        p21 = probr(spk, nta, r21, 2)
        p22 = probr(spk, nta, r22, 2)
        p41 = probr(spk, nta, r41, 4)
        p42 = probr(spk, nta, r42, 4)
        p43 = probr(spk, nta, r43, 4)
        p44 = probr(spk, nta, r44, 4)

        h21 = -np.sum((p21*np.log2((p21+EPS))))
        h22 = -np.sum((p22*np.log2((p22+EPS))))
        h41 = -np.sum((p41*np.log2((p41+EPS))))
        h42 = -np.sum((p42*np.log2((p42+EPS))))
        h43 = -np.sum((p43*np.log2((p43+EPS))))
        h44 = -np.sum((p44*np.log2((p44+EPS))))
        h4 = (h41+h42+h43+h44)/4.
        h2 = (h21+h22)/2.
        n1 = np.sum(nta)
        n2 = np.sum(np.floor((nta/2.)))
        n4 = np.sum(np.floor((nta/4.)))
        h0 = lagrange3(np.array(np.hstack((1./n4, 1./n2, 1./n1))), np.array(np.hstack((h4, h2, hc0))), 0.)
        #%h0=(8*hc0-6*h2+h4)/3; %parabolic extrapolation
        #%h0=(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hd*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
        #%hst=(4*hd-h21-h22)/2; %linear extrapolation
    elif _switch_val == 2.:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Naive correction
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        bias0 = matdiv(np.sum((p > EPS))-1., np.dot(2.*ntr, np.log(2.)))
        n = np.dot(p, ntr)
        pp = p
        #%(n+1/ntr)./(ntr+1);
        t2 = np.sum(matdiv((pp > EPS-pp)**2./(pp+EPS), np.dot(8.*ntr**2., np.log(2.))))
        h0 = hc0+bias0+t2
        bias0
        t2
    elif _switch_val == 3:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Panzeri
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        R = bayescount(ntr, p)
        bias = matdiv(R-1., np.dot(2.*ntr, np.log(2.)))
        h0 = hc0+bias
    elif _switch_val == 4:
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Montemurro
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        idx = randperm(ntr)
        ntr2 = np.round((ntr/2.))
        r21 = idx[0:ntr/2.]
        r22 = idx[int(ntr/2.+1.)-1:]
        p21 = probr(spk, r21, M)
        p22 = probr(spk, r22, M)
        h21 = -np.sum((p21*np.log2((p21+EPS))))
        h22 = -np.sum((p22*np.log2((p22+EPS))))
        bias0 = matdiv(np.sum((p > EPS))-1., np.dot(np.dot(2.*ntr, ns), np.log(2.)))
        bias1 = matdiv(np.sum((p21 > EPS))-1., np.dot(np.dot(ntr, ns), np.log(2.)))
        bias2 = matdiv(np.sum((p22 > EPS))-1., np.dot(np.dot(ntr, ns), np.log(2.)))
        h21 = h21+bias1
        h22 = h22+bias2
        h2 = (h21+h22)/2.
        h00 = hc0+bias0
        h0 = lagrange2(np.array(np.hstack((1./ntr2, 1./ntr))), np.array(np.hstack((h2, h00))), 0.)
        #%h0=(4*h00-h21-h22)/2;
    elif _switch_val == 5.:
        #%first recover absolute freqs
        n = np.round(np.dot(p, ntr))
        [h0] = saddleentr3(n)
    elif _switch_val == 6.:
        #%first recover absolute freqs
        n = np.round(np.dot(p, ntr))
        K = M ** L
        [h0] = simps_quad(n, betac[int(K)-1,:])
    elif _switch_val == 8:
        #%_range=range_shuffle(nt);
        r21 = range_frac(_range, nta, 2., 1.)
        r22 = range_frac(_range, nta, 2., 2.)
        p21 = probr(spk, nta, r21, 2.)
        p22 = probr(spk, nta, r22, 2.)
        h21 = -np.sum((p21*np.log2((p21+EPS))))
        h22 = -np.sum((p22*np.log2((p22+EPS))))
        h2 = (h21+h22)/2.
        n1 = np.sum(nta)
        n2 = np.sum(np.floor((nta/2.)))
        h0 = lagrange2(np.array(np.hstack((1./n2, 1./n1))), np.array(np.hstack((h2, hc0))), 0.)
        #%  case 7
        #%  % the following is just an example
        #% N=ntr*ns;
        #% lin_vec=[1:1:N];
        #% a_vec(1)=0;
        #% a_vec(2:1:N+1) = [-(lin_vec/N).*log(lin_vec/N)];
        #% clear lin_vec;
        #% [bias_value,var_bound]=bub_bv_func(a_vec,p,0); % this provide things in nats
        #% clear a_vec;
        #% bias_value=bias_value/log(2);
        #% %var_bound=var_bound/(log(2)*log(2));
        #% h0=hc0-bias_value;
        #% 
        #% case 8
        #%     
        #% N=ntr*ns;
        #% R=length(p); % R is the size of the response space - if you have not defined it yet, it could be determined from the size of the probability vector as I did on the left 
        #% lin_vec=[1:1:N];
        #% a_me_vec(1)=0;
        #% a_me_vec(2:1:N+1) = [-(lin_vec/N).*log(lin_vec/N)];
        #% a_mm_vec(1) = -1/(2*N*R);
        #% a_mm_vec(2:1:N+1) = [(-(lin_vec/N).*log(lin_vec/N)) + (1-(1/R))/(2*N) ];
        #% clear lin_vec;
        #% [bias_value_me,var_bound_me]=bub_bv_func(a_me_vec,p,0); % this provide things in nats
        #% [bias_value_mm,var_bound_mm]=bub_bv_func(a_mm_vec,p,0); % this provide things in nats
        #% clear a_me_vec,a_mm_vec;
        #% bias_value_me=bias_value_me/log(2);
        #% bias_value_mm=bias_value_mm/log(2);
        #% %IMPORTANT NOTE:
        #% %the bias_value_me should be subtracted from the raw entropy estimate
        #% % the bias_value_mm shold be subtracted from the naive-corrected entropy estimate (the one obtained from the raw estimate subtracting the C_1 Panzeri bias term computed with the "naive" counting of bins)
        #% bias0=(sum(p>eps)-1)/(2*ntr*ns*log(2));  
        #% h0=hc0+bias0;
        #%    
        #%h0=h0-bias_value_mm;
    
    #%error estimation, Latham's
    #%N=ntr;
    #%err=sqrt((sum(p.*log2(p+eps).^2)-(hd*L)^2)/(L*N));
    return h0

import unittest

import hx_test_utils as tst

class TestClass(unittest.TestCase):
  def test_hr(self):
        #for q in [2,1,0]:
        for L in [2,1]: #[5,2,1]:


            nta_arr=np.array([6*10,7*10,8*10])
            spk,L,nta,ns = tst._test_data_spk_singleval(L=L,nta_arr=nta_arr,value=1)
            #print spk.shape #1x2x8x3
            #print L,ns #2,3
            #print nta #
            assert spk.shape[1]==L
            #print type(spk[0,0,0,0]) #numpy.int16
            h_R=hr(spk,nta_arr,biastype=1)
            #print h_R
            h_R=hr(spk,nta_arr,biastype=0)
            #print h_R

            """
            range1=range_shuffle(nta)
            p,ctr=probr(spk,nta,range1,1, True)
            A = np.sort(ctr) # [0,0,0,...,0, ntr]
            assert A[len(A)-1] == sum(nta)
            assert sum(abs(A[range(len(A)-1)]))==0

            h_R=hr(spk,nta,biastype=BiasType.NAIVE0)
            print 'H(R)=',abs(h_R), '=?=' , EPS_PROB #3.20342650381e-16 2.22045e-16
            assert abs(h_R) < EPS_PROB*100

            """
            """
            tst
            nt=10
            spikes=np.zeros([nt,L])
            #print L,q,"L,q"
            pn, pd = pqmargs(spikes, L, q)
            #print pn
            #print pd
            self.assertEqual('foo'.upper(), 'FOO')

            spikes=np.zeros([nt,L])+1
            spikes[0,0]=0
            pn, pd = pqmargs(spikes, L, q)
            #print pn
            #print pd


            M=3
            nt=M**L
            spikes=n_ary(np.arange(0,M**L),L,M)
            assert spikes.shape == (nt,L)
            pn, pd = pqmargs(spikes, L, q)
            print pn
            print pd
            """




if __name__ == '__main__':
    unittest.main()
