
import numpy as np
import scipy
import matcompat
from probrs import probrs

import matplotlib.pylab as plt
from consts import EPS
from lagrange3 import lagrange3

def randperm(n):
    return np.random.permutation(range(n))

def hrs(spk, nt, biastype):
    # Local Variables: h44, r43, biastype, h2, h0, p22, h4, r44, bias, r41, h42, prs, r42, R, p44, p43, r22, r21, h22, bias1, h00, h21, p42, bias0, idx, bias2, p41, hnemt, ns, nt, K, ntr, M, L, hdt, h41, h43, hc2, hc3, hc0, hc1, betac, hc4, err, hc5, ntr4, ntr2, n, spk, t, p21
    # Function calls: lagrange3, simps_quad, log2, log, lagrange2, floor, max, sum, randperm, eps, hrs, saddleentr3, reshape, bayescount, round, probrs, size
    #%It will estimate the entropy of a binary chain giving the result in bits per bin
    #%trials=squeeze(spk(1,:,:))';
    #%Bias correction
    assert biastype<=1
    hc0 = 0.
    hc1 = 0.
    hc2 = 0.
    hc3 = 0.
    hc4 = 0.
    hc5 = 0.
    #%ntr=size(spk,3);
    ns = spk.shape[4-1] #matcompat.size(spk, 4.)
    L = spk.shape[2-1] #matcompat.size(spk, 2.)
    h0 = 0.
    err = 0.
    #M = 1.+matcompat.max(np.reshape(spk, 1., np.array([])))
    M = 1+max(spk.flatten()) #np.reshape(spk, 1., np.array([])))
    if M == 1:
        M = 2
    
    
    for t in range(1, (ns)+1):
            #%over all stimulus conditions

            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Direct estimation
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ntr = nt[int(t)-1]
            #%sohail
            if ntr<2:
                raise Exception('error: ntr<2')
            
            
            #%trials=(squeeze(spk(1,:,:,t)))'; % %trials set for current stimulus condition
            #prs = probrs(spk, np.array(np.hstack((np.arange(1., (ntr)+1)))), t, M)
            ii=np.array(   range(1, ntr+1), int)
            prs = probrs(spk, ii, t, M)
            #%it uses the same probr functions as Hr does!
            hdt = np.dot(-np.sum((prs*np.log2((prs+EPS)))), nt[int(t)-1])
            _switch_val=biastype
            if False: # switch 
                pass
            elif _switch_val == 0:
                bias = 0.
                h0 = h0+hdt
            elif _switch_val == 1:
                #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
                #%idx=randperm(ntr); %it performs a random permutation of the indeces to trials
                #%divide in ranges
                idx = randperm(ntr)
                ntr2 = np.floor((ntr/2.))
                ntr4 = np.floor((ntr/4.))
                r21 = idx[0:ntr2]
                r22 = idx[int(ntr2+1.)-1:2.*ntr2]
                r41 = idx[0:ntr4]
                r42 = idx[int(ntr4+1.)-1:2.*ntr4]
                r43 = idx[int(2.*ntr4+1.)-1:3.*ntr4]
                r44 = idx[int(3.*ntr4+1.)-1:4.*ntr4]
                p21 = probrs(spk, r21, t, M)
                p22 = probrs(spk, r22, t, M)
                p41 = probrs(spk, r41, t, M)
                p42 = probrs(spk, r42, t, M)
                p43 = probrs(spk, r43, t, M)
                p44 = probrs(spk, r44, t, M)
                h21 = -np.sum((p21*np.log2((p21+EPS))))
                h22 = -np.sum((p22*np.log2((p22+EPS))))
                h41 = -np.sum((p41*np.log2((p41+EPS))))
                h42 = -np.sum((p42*np.log2((p42+EPS))))
                h43 = -np.sum((p43*np.log2((p43+EPS))))
                h44 = -np.sum((p44*np.log2((p44+EPS))))
                h4 = np.dot(nt[int(t)-1], h41+h42+h43+h44)/4.
                h2 = np.dot(nt[int(t)-1], h21+h22)/2.
                h0 = h0+lagrange3(np.array(np.hstack((1./ntr4, 1./ntr2, 1./ntr))), np.array(np.hstack((h4, h2, hdt))), 0.)
                #%h0=h0+(8*hdt-6*h2+h4)/3;
                #%hs1=hs1+(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hdt*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
                #%hst=hst+(4*hdt-h21-h22)/2;
            elif _switch_val == 2.:
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #%Naive
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                bias0 = matdiv(np.sum((prs > EPS))-1., np.dot(2.*nt[int(t)-1], np.log(2.)))
                h0 = h0+hdt+np.dot(bias0, nt[int(t)-1])
            elif _switch_val == 3.:
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #%Panzeri
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                R = bayescount(nt[int(t)-1], prs)
                bias = R-1.
                h0 = h0+hdt+matdiv(np.dot(nt[int(t)-1], bias), np.dot(2.*nt[int(t)-1], np.log(2.)))
            elif _switch_val == 4.:
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                #%Montemurro
                #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                idx = randperm(ntr)
                #%it performs a random permutation of the indeces to trials
                #%divide in ranges
                ntr2 = np.round((ntr/2.))
                ntr4 = np.round((ntr/4.))
                r21 = idx[0:ntr2]
                r22 = idx[int(ntr2+1.)-1:]
                r41 = idx[0:ntr4]
                r42 = idx[int(ntr4+1.)-1:ntr2]
                r43 = idx[int(ntr2+1.)-1:3.*ntr4]
                r44 = idx[int(3.*ntr4+1.)-1:]
                p21 = probrs(spk, r21, t)
                p22 = probrs(spk, r22, t)
                h21 = -np.sum((p21*np.log2((p21+EPS))))
                h22 = -np.sum((p22*np.log2((p22+EPS))))
                bias0 = matdiv(np.sum((prs > EPS))-1., np.dot(2.*ntr, np.log(2.)))
                bias1 = matdiv(np.sum((p21 > EPS))-1., np.dot(ntr, np.log(2.)))
                bias2 = matdiv(np.sum((p22 > EPS))-1., np.dot(ntr, np.log(2.)))
                h21 = h21+bias1
                h22 = h22+bias2
                h2 = (h21+h22)/2.
                h00 = hdt+bias0
                #%   h0=h0+(4*h00-h21-h22)/2;    
                h0 = h0+lagrange2(np.array(np.hstack((1./ntr2, 1./ntr))), np.array(np.hstack((h2, h00))), 0.)
            elif _switch_val == 5.:
                #%Nemmenman
                #%first recover absolute freqs
                n = np.round(np.dot(prs, nt[int(t)-1]))
                [hnemt] = saddleentr3(n)
                h0 = h0+np.dot(hnemt, nt[int(t)-1])
            elif _switch_val == 6.:
                #%Nemmenman
                #%first recover absolute freqs
                n = np.round(np.dot(prs, nt[int(t)-1]))
                K = M ** L
                [hnemt] = simps_quad(n, betac[int(K)-1,:])
                h0 = h0+np.dot(hnemt, nt[int(t)-1])
            elif _switch_val == 8.:
                ntr = nt[int(t)-1]
                idx = randperm(ntr)
                ntr2 = np.floor((ntr/2.))
                r21 = idx[0:ntr2]
                r22 = idx[int(ntr2+1.)-1:2.*ntr2]
                p21 = probrs(spk, r21, t, M)
                p22 = probrs(spk, r22, t, M)
                h21 = -np.sum((p21*np.log2((p21+EPS))))
                h22 = -np.sum((p22*np.log2((p22+EPS))))
                h2 = np.dot(nt[int(t)-1], h21+h22)/2.
                h0 = h0+lagrange2(np.array(np.hstack((1./ntr2, 1./ntr))), np.array(np.hstack((h2, hdt))), 0.)
                #%h0=h0+(8*hdt-6*h2+h4)/3;
                #%hs1=hs1+(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hdt*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
                #% case 7
                #% lin_vec=[1:1:ntr];
                #% a_vec(1)=0;
                #% a_vec(2:1:ntr+1) = [-(lin_vec/ntr).*log(lin_vec/ntr)];
                #% clear lin_vec;
                #% [bias_value,var_bound]=bub_bv_func(a_vec,prs,0); % this provide things in nats
                #% clear a_vec;
                #% bias_value=bias_value/log(2);
                #% %var_bound=var_bound/(log(2)*log(2));
                #% h0=h0+hdt-bias_value;
                #%     
                #% case 8
                #%     
                #% N=ntr;
                #% R=length(prs); % R is the size of the response space - if you have not defined it yet, it could be determined from the size of the probability vector as I did on the left 
                #% lin_vec=[1:1:N];
                #% a_me_vec(1)=0;
                #% a_me_vec(2:1:N+1) = [-(lin_vec/N).*log(lin_vec/N)];
                #% a_mm_vec(1) = -1/(2*N*R);
                #% a_mm_vec(2:1:N+1) = [(-(lin_vec/N).*log(lin_vec/N)) + (1-(1/R))/(2*N) ];
                #% clear lin_vec;
                #% [bias_value_me,var_bound_me]=bub_bv_func(a_me_vec,prs,0); % this provide things in nats
                #% [bias_value_mm,var_bound_mm]=bub_bv_func(a_mm_vec,prs,0); % this provide things in nats
                #% clear a_me_vec,a_mm_vec;
                #% 
                #% bias_value_me=bias_value_me/log(2);
                #% bias_value_mm=bias_value_mm/log(2);
                #% %IMPORTANT NOTE:
                #% %the bias_value_me should be subtracted from the raw entropy estimate
                #% % the bias_value_mm shold be subtracted from the naive-corrected entropy estimate (the one obtained from the raw estimate subtracting the C_1 Panzeri bias term computed with the "naive" counting of bins)
                #% 
                #% bias0=(sum(prs>eps)-1)/(2*ntr*log(2));  
                #% h0=h0+hdt+bias0;
                #% h0=h0-bias_value_mm;
        
    h0 = h0 / np.sum(nt)
    return h0



import unittest

import hx_test_utils as tst

class TestClass(unittest.TestCase):
  def test_hr(self):
        #for q in [2,1,0]:
        for L in [2,1]: #[5,2,1]:


            nta_arr=np.array([6*10,7*10,8*10])
            spk,L,nta,ns = tst._test_data_spk_singleval(L=L,nta_arr=nta_arr,value=1)
            assert spk.shape[1]==L
            h_R=hrs(spk,nta_arr,biastype=1)
            print h_R
            h_R=hrs(spk,nta_arr,biastype=0)
            print h_R



if __name__ == '__main__':
    unittest.main()
