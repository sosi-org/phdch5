import numpy as np
import unittest
from range_frac import *

RESP_TYPE_INPUT=np.int16

def n_ary(twi, L,M):
    assert len(twi.shape)==1
    wi=np.zeros([len(twi),L],int)
    for j in range( len(twi) ):
        v = int(twi[j])
        for i in range(0, L):
            wi[j,L-1-i] = v % M #digit(i of twi[j])
            v=int(v/M)
        assert v==0
    return wi

def test__n_ary():
    L=3;M=2
    nt=M**L
    spikes=n_ary(np.arange(0,M**L),L,M)
    assert spikes.shape == (nt,L)
    #print spikes


def _checktype_nta(nta, ns):
    COUNT_TYPE = np.int64
    #also allow plain Python list of int
    assert type(nta) is np.ndarray
    assert type(nta[0]) is COUNT_TYPE
    #alsochecks the consistency
    assert len(nta) == ns



def _test_data_spk_singleval(L,nta_arr, value):

    L=L; nta=np.array(nta_arr,int);ns=len(nta)
    _checktype_nta(nta, ns)
    RESP_TYPE = int
    spk = np.zeros([1,L,max(nta),ns], RESP_TYPE_INPUT)
    for i in range(1):
        for l in range(L):
            for s in range(ns):
                #print ns, nta[s]
                for tr in range(nta[s]):
                    spk[i,l,tr,s] = RESP_TYPE_INPUT(value)
    return spk,L,nta,ns

def _test_data_spk_rand(L,nta_arr, M):
    nta=np.array(nta_arr,int);ns=len(nta)
    _checktype_nta(nta, ns)
    #spk = np.zeros([1,L,len(nta),ns], np.int8)
    spk = np.zeros([1,L,max(nta),ns], RESP_TYPE_INPUT)
    for i in range(1):
        for l in range(L):
            for s in range(ns):
                #print ns, nta[s]
                for tr in range(nta[s]):
                    spk[i,l,tr,s] = RESP_TYPE_INPUT(np.random.random_integers(0,M-1))
                    #M=max(spk.flatten())+1 ==> between [0,M-1]
    return spk,L,nta,ns


def _test_data_spk_plain(pa,L,nta_arr):
    M=len(pa)
    nta=np.array(nta_arr,int);ns=len(nta)
    _checktype_nta(nta, ns)
    spk = np.zeros([1,L,max(nta),ns], RESP_TYPE_INPUT)
    for i in range(1):
        for l in range(L):
            for s in range(ns):
                for tr in range(nta[s]):
                    #spk[i,l,tr,s] = RESP_TYPE(np.random.choice(range(M),p=pa))
                    if np.random.random_integers(0,1)==0:
                        #R = 1 # np.random.choice(range(M),p=pa)
                        R = np.random.choice(range(M),p=pa)
                    else:
                        #R = 0 #
                        R = s
                    spk[i,l,tr,s] = RESP_TYPE_INPUT(R)


    return spk,L,nta,ns




def _test_data_spkNtL___allcases(L, M):
    """ Nt x L """
    ns=3
    nt=M**L
    w=np.arange(0,M**L)
    spikes=n_ary(w,L,M)
    assert spikes.shape == (nt,L)
    return spikes

def _test_data_spk___allcases(L, M):
    ns=3
    nt=M**L
    w=np.arange(0,M**L)
    spikes=n_ary(w,L,M)
    assert spikes.shape == (nt,L)

    #ns=len(nta_arr)
    spk=np.tile(spikes.T.reshape((1,L,nt,1)),   (1,1,1,ns))
    nta_arr=np.array([nt]*ns)
    #print spikes.shape
    return spk,L,nta_arr,ns


"""
def _test_data_spk_rand_pa(pa,L,nta_arr, M):
    #todo: based on _test_data_spk_rand
    nta=np.array(nta_arr,int);ns=len(nta)
    _checktype_nta(nta, ns)
    spk = np.zeros([1,L,max(nta),ns], RESP_TYPE_INPUT)
    for i in range(1):
        for l in range(L):
            for s in range(ns):
                for tr in range(nta[s]):
                    spk[i,l,tr,s] = 0 #RESP_TYPE(np.random.random_integers(0,M-1))***
                    raise Exception("Not implemented")
    return spk,L,nta,ns
"""

class TestX(unittest.TestCase):
  def test_test_data_gen(self):
        for ns in [2,5]:
            nta_arr=[10]*ns #todo: different
            #for q in [2,1,0]:
            for M in [2,5]:
                for L in [5,2,1]:
                    pa = np.array([0.1]*ns)
                    pa = pa / sum(pa)
                    assert sum(pa)==1.0
                    #z=_test_data_spk_rand_pa(pa,L,nta_arr, M)
                    z=_test_data_spk_plain(pa,L,nta_arr)
                    value=2
                    z=_test_data_spk_singleval(L,nta_arr, value)
                    z=_test_data_spk_rand(L,nta_arr, M)




if __name__ == '__main__':
    unittest.main()
