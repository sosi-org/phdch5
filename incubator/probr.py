
import numpy as np
import scipy
import matcompat

#import matplotlib.pylab as plt
import math
from consts import EPS, BIG_EPS
from mytypes import *
from range_shuffle import range_shuffle
from range_frac import range_frac

def probr(spk,nta,r,f, return_count=False):
    #function p=probr(spk,nt,r,f)
    """
    nta and f together determin the trial numbers. since nta is th eoriginal number of trials, f is also needed for a subset of trials.
    The function probrs does not have any of of nta and f as arguments and seems simpler. Because if focuses on one stim only.

    :param spk:
    :type spk: np.ndarray
    :param nta:
    :type nta: list
    :param r:  The order of trials.
                #r= in what order (trailed by zero)
    :type r:
    :param f: an integer that the number of trials is divided by it.
                    #f= divide by what integer number

    :type f: int
    :return: p
    :rtype:
    this function works with the static version
    f is the factor that divides the number of trials to use
    nta stores the number of trials per stimulus condition
    """
    assert r.size>0
    DEBUG=True
    if DEBUG:
        _ns= spk.shape[3]
        big_ntr=max(nta)
        assert   big_ntr==spk.shape[2]
        assert r.shape[1] == spk.shape[2]/f
        #print r.shape, (_ns,big_ntr/f), f
        assert r.shape==(_ns,big_ntr/f)
        #assert sorted(r)
        ns0=len(nta)
        for si in range(0,ns0):
            #r[si].sort()
            #print r[si]
            #print r.shape
            n1=nta[si]

            #print r[si,:]
            #invar:  r.shape[1] == spk.shape[2]/f
            #print r.shape[1],nta[si],spk.shape[2],f #40,60,80,2  #but 30 are filled in.
            # 80,60,80,1

            #print  r.shape[1],n1/f,f,n1,"*"  #20 17 4 70
            #assert r.shape[1]>=n1 #not guaranteed. Instead:
            #assert r.shape[1]<=n1/f #not guaranteed
            #print r.shape[1],n1/f #n1/f out of s.shape[1] are used.  always larger.
            assert r.shape[1]>=n1/f #***
            n2=n1/f
            assert r.shape[1]>=n2
            assert sum(abs(r[si,n2:]))==0 #all others are zero
            q=r[si,:n2].copy()
            #print q
            #print len(q),n2  #always equal
            assert len(q) == n2
            #q.sort()
            #print "q(unsorted)=",q
            #print min(q)  #but it should not contain zero!!
            #if min(q)<1:  #has no zeros
            #    print "warning: min(q) is %d "%(min(q),)
            assert not min(q)<1
            #min(q) can be zero
            #assert min(q)==0   #1 may start from 30
            q.sort() #leaves r unsorted
            #print "unsorted=",r[si,:n2]
            #assert q==np.arange(0,nta[si])
            #print "q(sorted)="
            #print q
            #assert sum(abs(q-np.arange(0,n2)))==0   #it may start from 30. no quarantee about the contents of r
            #print si
            #print r.shape
            #print r.size
            #print r[si,:]
            #print f #4
            #assert max(r[si,:])<nta[si]
            assert max(r[si,:])<=nta[si]
            #Statement: r[si,:] will contain nta[si]/f elements. The rest are zero. These elements are valid elements of spk[0,,,si], i.e. spk[0,?,:nta[si],si].


    #rsa  LxNxS  1xLxNxS  1LNS spk1xLxNxS spk1LNS
    assert type(spk) is np.ndarray
    #_checktype_spk1LNS(spk, nta=None)
    #assert len(spk.shape)==4
    ntr=spk.shape[3-1] #ntr=size(spk,3);
    ns=spk.shape[4-1] #ns=size(spk,4);
    L=spk.shape[2-1] #size(spk,2);
    M=max(spk.flatten())+1 #M=max(reshape(spk,1,[]))+1;
    #_checktype_nta(nta,ns)

    #print type(f)
    #%number of trials for the subset
    assert type(f) in [COUNT_TYPE,int]
    #todo: some trials are removed in QE
    new_nta = nta/f #new_nt=floor(nt/f);   #:type new_nta: list
    assert len(new_nta)==len(nta)
    #_checktype_nta(nta,ns/f)

    #%these must be selected randomly from the original set
    tot_nt=sum(new_nta) #tot_nt=sum(new_nt);
    trials=np.zeros([tot_nt,L], COUNT_TYPE) #trials=zeros(tot_nt,L);
    i=1 #i=1;
    if L>1:
        for s in range(ns): #for s=1:ns
            #trials(i:i+new_nta(s)-1,:)=squeeze(spk(1,:,r(s,1:new_nta(s)),s))';

            #print new_nta[s],L
            #print r[s,range(new_nta[s])]-1
            #print spk[0,:,r[s,range(new_nta[s])]-1,s]
            #print trials[(i-1):(i-1+new_nta[s]-1+1),:]
            #print trials[(i-1):(i-1+new_nta[s]-1+1),:].shape

            #trials[(i-1):(i-1+new_nta[s]-1+1),:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s] ) #.transpose()?
            #trials[(i-1):(i-1+new_nta[s]-1+1),:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s], [0,3]) #.transpose()?
            #trials[(i-1):(i+new_nta[s]-1),:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s], [0,3]) #.transpose()?
            ri = r[s,range(new_nta[s])]-1
            trials[(i-1):(i+new_nta[s]-1),:] = np.squeeze( spk[0,:,ri,s], [0,3]) #.transpose()?
            assert min(ri)>=0 #numpy?!!!


            #   #idx[s,0:(nta[s]-1)] = np.random.permutation(range(nta[s]))
            #   idx[s,0:nta[s]] = np.random.permutation(range(nta[s]))+1

            #i=i+new_nta(s);
            i += new_nta[s]
        #end
    else:
        #r[s,range(new_nta[s])] ==?
        for s in range(ns): #for s=1:ns
            #trials(i:i+new_nta(s)-1,:)=squeeze(spk(1,:,r(s,1:new_nta(s)),s));
            #trials[i-1:i+new_nta(s)-1+1,:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s] , [0,3] ) #no transpose
            #print new_nta[s]
            #print trials[(i-1):(i+new_nta[s]-1),0].shape
            #A=spk[0,:,r[s,range(new_nta[s])]-1,s]
            #print type(s) #int
            #print type(r) #ndarray
            #print r.shape #1428 x 16
            #print r
            #ri=r[s][0:new_nta[s]]-1
            ri=r[s, 0:new_nta[s]]-1
            #print ri # 16x16?
            #print new_nta.shape #[1428]! [16,16,....,16]
            #print ri.shape #(16,)
            #r is a list
            #print type(r), type(r[0])  # a list of nd array???

            #print type(ri)  #np.ndarray
            A=spk[0,:,ri,s]
            #A=spk[0,:,r[s, 0:new_nta[s]]-1,s]
            assert min(ri)>=0 #numpy?!!!!


            #print A.shape #(1000,1)
            A2=np.squeeze(A) #A2=np.squeeze( A , [0,3] )
            trials[(i-1):(i+new_nta[s]-1),0] = A2 #no transpose
            #      ===               ==                                                ==
            #same!!!
            i+=new_nta[s]
            #i=i+new_nta(s);
        #end
    #end

    if False:
        #M_pow_L = math.pow(M,L)
        #p=zeros(1,M^L);
        p = np.zeros([1,math.pow(M,L)], PROB_TYPE)
        #count=zeros(M^L,1);
        count=np.zeros([math.pow(M,L),1],COUNT_TYPE)
    #wi=1+trials*(M.^[0:L-1])';
    #wi=1+trials*np.power(M,[0:L-1]) #'; tranpose
    #print "-------"
    #print trials.shape #6x2
    #print np.power(M,range(L)).shape #2x-
    #print trials #6x2  [1,2]
    #print np.power(M,range(L)) #2x-   [[1,1], x 6]
    B = np.power(M,np.array(range(L)).reshape([L,1]))  #'; tranpose
    #print B.shape #2x-   [[1,1], x 6]
    #print B #2x-   [[1,1], x 6]
    #wi=1+trials .dot(np.power(M,range(L))) #'; tranpose
    # trials: 2dim
    wi=1+trials .dot(B)
    #print 'min=',min(min(wi))
    #assert min(min(wi))>=1
    #assert max(max(wi))>=1

    #print _debug_show_table(np.sort(wi.flatten()),range(100,200)) #
    #print _debug_show_table2(wi.flatten()) #new_nta
    #print new_nta
    #print wi.shape  # 6x1   NTR x
    assert wi.shape[0] == sum(new_nta)
    assert wi.shape[1] == 1
    #print math.pow(M,L)
    #print max(max(wi))-1,math.pow(M,L)
    #assert max(max(wi))-1+EPS < math.pow(M,L)
    assert max(max(wi))-1 < math.pow(M,L) +BIG_EPS
    assert min(min(wi))-1 >= 0  #min >= 1

    assert max(max(wi)) < math.pow(M,L)+1 +BIG_EPS

    #wi : NT x 1 #max<=(M^L)
    #count=histc(wi,[1:M^L+eps]);
    #edges = np.array(range(1+int(EPS+math.pow(M,L))))+EPS #bin edges, including the rightmost edge,
    #edges = np.array(range(0,1+int(EPS+math.pow(M,L))))+EPS #bin edges, including the rightmost edge,
    maxrange = 1+(BIG_EPS+math.pow(M,L))
    #print max(max(wi)),maxrange # 4,5.0  or 25 26.0 #25 26.000001
    edges = np.array(range(0,int(maxrange)))+0.1 #EPS does not work here!! #bin edges, including the rightmost edge,
    #If you start the range from 1, items will be missing

    #print edges
    #print _debug_show_table2(wi.flatten()) #
    #todo: write an integer historam in C (or use a library)
    count,e2 = np.histogram(wi.flatten(), edges) #
    assert sum(count) == sum(new_nta)

    #print count.astype(COUNT_TYPE)
    #print sum(count.astype(COUNT_TYPE)), sum(nta)
    #print '======',sum(nta)

    #print count.shape # (3,)
    #print e2 #???
    #print wi
    #print count

    #p=(count'/sum(count));
    #p=(count.transpose()/sum(count))
    p=count / PROB_TYPE(sum(count))
    #print p

    assert p.shape==(M**L,)

    if return_count:
        count=count.astype(COUNT_TYPE)
        return p,count
    else:
        return p





"""
def probr_NOT(spk, nt, r, f):

    # Local Variables: tot_nt, count, new_nt, f, i, M, ntr, spk, L, wi, p, s, r, trials, ns, nt
    # Function calls: floor, max, sum, eps, zeros, reshape, histc, squeeze, probr, size
    #%this function works with the static version 
    #%f is the factor that divides the number of trials to use
    #%nt stores the number of trials per stimulus condition
    ntr = matcompat.size(spk, 3.)
    ns = matcompat.size(spk, 4.)
    L = matcompat.size(spk, 2.)
    #M = matcompat.max(np.reshape(spk, 1., np.array([])))+1.
    M = max(spk.flatten())+1.
    #%number of trials for the subset
    new_nt = (np.floor((np.array(nt) / f)))
    #%these must be selected randomly from the original set
    tot_nt = np.sum(new_nt)
    trials = np.zeros((tot_nt, L), int)
    i = 1
    if L > 1.:
        for s in np.arange(1., (ns)+1):
            trials[int(i)-1:i+new_nt[int(s)-1]-1.,:] = np.squeeze(spk[0,:,r[int(s)-1,0:new_nt[int(s)-1]],int(s)-1]).conj().T
            i = i+new_nt[int(s)-1]
            
    else:
        for s in np.arange(1., (ns)+1):
            #trials[int(i)-1:i+new_nt[int(s)-1]-1.,:] = np.squeeze(spk[0,:,r[int(s)-1,0:new_nt[int(s)-1]],int(s)-1])
            i = i+new_nt[int(s)-1]
            
        
    
    p = np.zeros((1, (M ** L)),int)
    count = np.zeros([(M ** L), 1],int)
    wi = 1.+np.dot(trials, (M**np.array(np.hstack((np.arange(0., (L-1.)+1))))).conj().T)
    count = histc(wi, np.array(np.hstack((np.arange(1., ((M ** L)+EPS)+1)))))
    p = matdiv(count.conj().T, np.sum(count))
    return p
"""




import hx_test_utils as tst
import unittest

class Tests_probs(unittest.TestCase):
  #todo: generate_spka
  def test_1(self):
        for L in [5,2,1]:
            for typ in [1,2,3]:
                if typ==1:
                    spk,_L,nta_arr,_ns = tst._test_data_spk___allcases(L=L,M=3)
                elif typ==2:
                    spk,_L,nta_arr,_ns = tst._test_data_spk___allcases(L=L,M=3)
                    spk=spk*0+1
                elif typ==3:
                    _nta_arr = np.array([6*10,7*10,8*10])
                    spk,_L,nta_arr,_ns = tst._test_data_spk_rand(L=L, nta_arr=_nta_arr, M=3)
                #todo: more tests
                else:
                    assert False
                for f in [1,4]:
                    if min(nta_arr)>=f:
                        for k in range(1,f+1):
                            #print "nta_arr",nta_arr, "f=",f,"k=",k, "typ",typ, "L",L
                            _range = range_shuffle(nta_arr)
                            r43 = range_frac(_range, nta_arr, f, k)
                            p43 = probr(spk, nta_arr, r43, f, return_count = False )
                            #print p43, '=>',sum(p43)
                            assert abs(sum(p43.flatten()) - 1.0) < EPS
                            #print "p.shape",p43.shape
                            #todo: check the resulting values



if __name__ == '__main__':
    unittest.main()
