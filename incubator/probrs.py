
import numpy as np
import scipy
import matcompat
import matplotlib.pylab as plt
from consts import EPS 

def probrs(spk, r, s, M):
    #r selects the trials
    #r will also determines the number of trials
    #Unlike prob.py, M is explicitly provided because at one stim may have a smaller M and we cannot derive M. in prob.py, M can be deribved frmo data and it doesnt make any difference if the maximum responses is smaller than the intended M.

    assert len(r.shape)==1
    _numtrial = len(r)  #PLANGNOTE: annotation
    assert s>=1
    assert type(s) is int, "%r"%(type(s))

    DIM_STIM=3
    DIM_AUX=0
    DIM_L=1
    # Local Variables: spkt, count, M, ntr, spk, L, wi, p, s, r, trials, ns
    # Function calls: sum, eps, length, zeros, histc, squeeze, probrs, size
    #%this function works with the static version 
    #%f is the factor that divides the number of trials to use
    #ntr = length(r)
    ntr = len(r)
    #ns = spk.shape[4-1] # matcompat.size(spk, 4.)
    #L = spk.shape[2-1] #matcompat.size(spk, 2.)
    ns = spk.shape[DIM_STIM]
    L = spk.shape[DIM_L]

    #spkt=squeeze(spk(1,:,r,s));
    #if (L==1)
    #    spkt=spkt';
    #end


    #spkt = np.squeeze(spk[0,:,int(r)-1,int(s)-1])
    #print type(s) #int
    assert type(s) is int
    #print type(r[0]) #float
    assert type(r[0]) in [np.int64,np.int32,np.int16,np.int8]
    #spkt = np.squeeze(spk[0,:,int(r)-1,int(s)-1])
    #spkt = np.squeeze(spk[0,:,r-1,int(s)-1])
    #if L == 1.:
    #    spkt = spkt.T

    #spkt = np.squeeze(spk[0,:,r-1,int(s)-1],axis=(DIM_AUX,DIM_STIM,)) #cancel out (0,3,) ; (-,stim)
    spkt0 = spk[0,:,r-1,int(s)-1]
    assert min(r-1)>=0
    assert len(spk.shape)==4
    #assert spk.shape==(1,?,?,)
    assert spk.shape[0]==1

    #print spk.shape ##1x1x16x1428   1x2x16x714
    #print r.shape #16
    #print spkt0.shape #16 x 1    16x2
    #why???
    #big_ntr = spk.shape[2]
    assert spkt0.shape == (ntr,L) #why?????

    #spkt = np.squeeze(spkt0,axis=(DIM_AUX,DIM_STIM,)) #cancel out (0,3,) ; (-,stim)
    spkt = spkt0.T
    assert spkt.shape == (L,ntr)
    #if L == 1.:
    #    spkt = spkt.T


    
    #trials=spkt';
    #p=zeros(1,M^L);
    #count=zeros(M^L,1);
    #wi=1+trials*(M.^[0:L-1])';
    #count=histc(wi,[1:M^L+eps]);
    #if count==0
    #   % 'count==0'
    #    p=count'/1;
    #else
    #    p=(count'/sum(count));
    #end

    #print spkt.shape

    # size([1:3]) %   1   3

    #%trials=(reshape(spkt,L,[]))';
    #trials = spkt.reshape([]) #spkt.conj().T
    trials = spkt.T
    assert trials.shape == (ntr,L)

    p = np.zeros([1, M**L], int)
    count = np.zeros([M**L, 1], int)
    #wi=1+trials*(M.^[0:L-1])';   # [1:L]  : 1xL   [1:L]' : Lx1
    #wi = 1.+np.dot(trials, (M**np.array(np.hstack((np.arange(0., (L-1.)+1))))).conj().T)
    BB=M ** np.arange(0, L-1+1).reshape([L,1])  #L x 1

    #assert BB.shape == ()
    #print BB.shape  #2
    #print trials.shape #2x16
    wi = 1+np.dot(trials, BB)   # ntr x L *  Lx1
    #count=histc(wi,[1:M^L+eps]);

    #count = histc(wi, np.array(np.hstack((np.arange(1., ((M ** L)+EPS)+1)))))

    #todo: myhist.myhistint(wi, M**L)
    maxrange=int(M ** L +0.0001)
    edges = np.array(range(0,maxrange+1)) + 0.01
    count,e2 = np.histogram(wi.flatten(), edges) #
    #print count, e2
    #[0 0 0 0 0 3 2 4 0 1 0 2 0 0 1 3] [  1.00000000e-02   1.01   2.01   3.01  4.01000000e+00   5.01000000e+00   6.01000000e+00   7.01000000e+00

    assert sum(count)==wi.size


    assert count.shape == (M**L,)
    #if count == 0:  #causes error
    #    #% 'count==0'
    #    p = count.conj().T/1.
    #    assert False, "Not checked"
    #else:
    if 1:
        #p = (count.conj().T) / float(np.sum(count))
        #p = (count.T) / float(np.sum(count))
        assert count.shape==(M**L,)
        p = (count.T) / float(np.sum(count))

    return p







import hx_test_utils as tst
import unittest

class Tests_probs(unittest.TestCase):

  def test_1(self):
        M=3
        for L in [5,2,1]:
            for typ in [1,2,3]:
                if typ==1:
                    spk,_L,nta_arr,_ns = tst._test_data_spk___allcases(L,M)
                    #print spk.shape, "*"
                elif typ==2:
                    spk,_L,nta_arr,_ns = tst._test_data_spk___allcases(L,M)
                    spk=spk*0+1
                elif typ==3:
                    _nta_arr = np.array([6*10,7*10,8*10])
                    spk,_L,nta_arr,_ns = tst._test_data_spk_rand(L=L, nta_arr=_nta_arr, M=M)
                #todo: more tests
                else:
                    assert False
                if True:
                            #print typ,spk.shape
                            ns=spk.shape[3]
                            prs_whole=np.zeros((M**L,ns),float)
                            #print "nta_arr",nta_arr, "L=",L, "M=",M,"typ",typ
                            for si in range(0,ns):
                                #_range = range_shuffle(nta_arr)
                                #r43 = range_frac(_range, nta_arr, f, k)
                                def randperm(n):
                                    return np.random.permutation(range(n))+1

                                idx = randperm(nta_arr[si])
                                r43 = idx[:]
                                #r43= idx[:len(idx)/2]
                                p43 = probrs(spk, r43, si+1, M)
                                #assert prs_whole.shape[0]==len(p43)

                                #print si, prs_whole.shape , p43.shape, type(p43[0]), type(prs_whole[0,0])
                                #print prs_whole[:,si], "[:,si]1"
                                prs_whole[:,si] = p43 #does not work, if LHS is int and RHS is float.
                                #print prs_whole[:,si], "[:,si]2"

                                assert abs(sum(p43.flatten()) - 1.0) < EPS
                                #print "p.shape",p43.shape
                                #print p43, "p43"
                            #print prs_whole.T,"*"
                            #print "=>",sum(prs_whole)
                            #print "=>",np.sum(prs_whole, axis=0)
                            assert sum(abs(np.sum(prs_whole, axis=0)-1.0))<EPS
                            #print spk, nta_arr
                            #todo: check the resulting probability values


if __name__ == '__main__':
    unittest.main()
