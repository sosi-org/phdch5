#todo: not processed
#FIXME: make a unittest
import numpy as np
import scipy
import matcompat
import matplotlib.pylab as plt
from consts import EPS

def randperm(n):
    return np.random.permutation(range(n))+1

def q_shuffle(spk, nta, q):

    # Local Variables: memd, Igs, g, spks, new_spike, k, memorys, M, L, q, spk, qe, t, memory, memds, ns, nta, Ig, rbin
    # Function calls: q_shuffle, max, randperm, length, zeros, reshape, squeeze, find, size
    #%does a shuffling preserving marginals up to order q
    #%ntr=size(spk,3);
    ns = spk.shape[3]  #matcompat.size(spk, 4.)
    L = spk.shape[1] #matcompat.size(spk, 2.)
    #spks = np.zeros(matcompat.size(spk), int)
    spks = np.zeros(spk.shape, int)
    for t in range(1, ns+1):
        #%for t=1:ns
        #%over all stimulus conditions
        assert type(t) is int
        rbin = randperm(nta[t-1])
        #M = 1.+matcompat.max(np.reshape(spk[0,:,:,int(t)-1], 1., np.array([])))
        M = 1+max(spk[0,:,:,t-1].flatten())
        #spks[0,0,0:nta[int(t)-1],int(t)-1] = spk[0,0,int(rbin)-1,int(t)-1]
        #spks[0,0, 0:nta[t-1] ,t-1] = spk[0,0,rbin-1,t-1]
        A=spk[0,0,rbin-1,t-1]
        spks[0,0, 0:nta[t-1] ,t-1] = A
        spks[0,0, 0:nta[t-1] ,t-1] = spk[0,0,rbin-1,t-1]
        #%first bin comes from random picking of first bins
        #%for k=2:L   %now we have to choose one by one the next bins
        #for k in np.arange(2., (L)+1):
        for k in range(2, L+1):
            #%now we have to choose one by one the next bins

            if q > k-1:
                qe = k-1
            else:
                qe = q


            #memory=squeeze(spk(1,k-qe:k-1,1:nta(t),t))'; %past spikes
            #memorys=squeeze(spks(1,k-qe:k-1,1:nta(t),t))';
            #if (qe==1)
            #    memory=memory';
            #    memorys=memorys';
            #end
            #memd=memory*[M.^[0:qe-1]]';
            #memds=memorys*[M.^[0:qe-1]]';



            #memory=squeeze(spk(1,k-qe:k-1,1:nta(t),t))'; %past spikes
            #memorys=squeeze(spks(1,k-qe:k-1,1:nta(t),t))';

            ss0= spk[0:1,(k-qe-1):(k-1),0:nta[t-1],(t-1):t] # 1x (k-1)x (ntx) x 1
            ss1=spks[0:1,(k-qe-1):(k-1),0:nta[t-1],(t-1):t] #

            #ss0= spk[0,(k-qe-1):(k-1),0:nta[t-1],t-1] # 1x (k-1)x (ntx) x 1
            #ss1=spks[0,(k-qe-1):(k-1),0:nta[t-1],t-1] #
            #ss0= spk[[0],np.arange((k-qe-1),(k-1)),np.arange(0,nta[t-1]),t-1] # 1x (k-1)x (ntx) x 1
            #ss1=spks[[0],np.arange((k-qe-1),(k-1)),np.arange(0,nta[t-1]),t-1] #
            #print spk.shape, spks.shape
            #print t
            assert len(ss0.shape)==4


            #xxx= spk[0:1] #(5, 243, 3)
            #print xxx.shape
            #xxx= spk[0:1][(k-qe-1):(k-1)]  #(0, 243, 3)  !!
            #print xxx.shape
            #xxx= spk[0:1][(k-qe-1):(k-1)][0:nta[t-1]] #(5, 243, 3)
            #print xxx.shape
            #xxx= spk[0:1][(k-qe-1):(k-1)][0:nta[t-1]][t-1] #(5, 243, 3)   
            #print xxx.shape


            #ss0= spk[0:0][(k-qe-1):(k-1)][0:nta[t-1]][t-1] # 1x (k-1)x (ntx) x 1
            #ss1=spks[0:0][(k-qe-1):(k-1)][0:nta[t-1]][t-1] #
            #print ss0.shape,"===================="
            #print ss1.shape
            #ss0_squeeze=np.squeeze(ss0) #wont work if qe==1            
            #ss1_squeeze=np.squeeze(ss1)
            #print "qe=",qe
            #print "****", ss0.shape
            #print "****", ss1.shape
            ss0_squeeze=np.squeeze(ss0,axis=(0,3)) #wont work if qe==1            
            ss1_squeeze=np.squeeze(ss1,axis=(0,3))
            #print "L=",L, ss0_squeeze.shape , (   (k-1)-(k-qe-1), nta[t-1] )

            assert ss0_squeeze.shape == (   (k-1)-(k-qe-1), nta[t-1] )
            assert ss1_squeeze.shape == (   (k-1)-(k-qe-1), nta[t-1] )
            memory = ss0_squeeze.T
            #memory = ss0.reshape(ss0.).conj().T
            #%past spikes
            memorys = ss1_squeeze.T
            assert memory.shape == (   nta[t-1], (k-1)-(k-qe-1),  )
            assert memorys.shape == (   nta[t-1], (k-1)-(k-qe-1) )
            #memorys = ss1.reshape(ss1).conj().T
            assert (k-1)-(k-qe-1) == qe
            assert memory.shape == (   nta[t-1], qe,  )
            assert memorys.shape == (   nta[t-1], qe )
            #todo: dont .T but do .dot() in the opposite order.
            #if qe == 1:
            #    memory = memory.conj().T
            #    memorys = memorys.conj().T

            #FIXME
            #memd = np.dot(memory, np.array(np.hstack((M**np.array(np.hstack((np.arange(0, (qe-1.)+1))))))).conj().T)
            #memds = np.dot(memorys, np.array(np.hstack((M**np.array(np.hstack((np.arange(0, (qe-1.)+1))))))).conj().T)
            #ppp0= M**np.array(range(0, qe-1+1),int).reshape((1,qe)).T
            #ppp1 = M**np.array(range(0, qe-1+1),int).conj().T
            #print "qe =",qe

            #print (M**np.arange(0, qe-1+1)), "----"
            #print (M**np.arange(0, qe-1+1)).shape
            #ppp0= (M**np.arange(0, qe-1+1)).reshape((qe-1,1))
            ppp0= (M**np.arange(0, qe-1+1)).reshape((qe,1))
            #print memory.shape, ppp0.shape
            #print memorys.shape, ppp0.shape
            assert memory.shape[1]==ppp0.shape[0] #many times N x 0 , 0 x m
            assert memorys.shape[1]==ppp0.shape[0]
            #print ppp0.shape
            if qe>0: #may be zero
                assert type(ppp0[0,0]) in [int, np.int8,np.int16,np.int32,np.int64]

            memd = memory.dot(ppp0)
            memds = memorys.dot(ppp0)
            if qe>0: #may be zero
                assert type(memd[0,0]) in [int, np.int8,np.int16,np.int32,np.int64]
            assert memd.shape == (nta[t-1],1)
            assert memds.shape == (nta[t-1],1)

            for g in range(0, M**qe-1+1):

                #Igs=find(memds==g);
                #Ig=find(memd==g);

                AA=((memds - g)==0).flatten()
                #print AA
                #print AA.shape
                Igs_1 = np.nonzero(AA)

                Igs_1 = np.nonzero(((memds - g)==0).flatten())[0] # +1
                Ig_1 = np.nonzero(((memd - g)==0).flatten())[0] # +1
                assert len(Ig_1)==len(Igs_1)
                #print Ig_1
                import sys


                #new_spike = spk[0,k-1,Ig-1,t-1].flatten()
                new_spike = spk[0,k-1,Ig_1,t-1].flatten()
                sys.stdout.flush()
                #print "*",Ig_1,"*"
                #print type(Ig_1)
                #print (Ig_1.shape)
                #print new_spike.shape,(len(Ig_1),)
                assert new_spike.shape==(len(Ig_1),)
                #%mix them
                #new_spike = new_spike[int(randperm(len(Ig)))-1]
                #new_spike = new_spike[randperm(len(Ig_1))-1] #Apply the shuffle
                rr=randperm(len(Ig_1))-1
                #print rr
                #new_spike = new_spike[rr] #Apply the shuffle
                #if len(rr)>0:
                #    new_spike = new_spike[rr]
                #else:
                #    new_spike = np.arange([1])[0:0]

                if len(rr)>0:
                    new_spike = new_spike[rr]


                    #%put them back
                    #spks[0,k-1,Igs-1,t-1] = new_spike
                    spks[0,k-1,Igs_1,t-1] = new_spike
                    #why works
            #%end

    return spks

"""
    for g=0:M^qe-1
        Igs=find(memds==g);
        Ig=find(memd==g);
        new_spike=spk(1,k,Ig,t);
        %mix them
        new_spike=new_spike(randperm(length(Ig)));
        %put them back
        spks(1,k,Igs,t)=new_spike;
    end
"""

#spk_q = q_shuffle(spk2_dig, nts, q)


import hx_test_utils as tst
import unittest

def generate_spka(L,M, typ):
    if typ==1:
        spk,_L,nta_arr,_ns = tst._test_data_spk___allcases(L,M)
        #print spk.shape, "*"
    elif typ==2:
        spk,_L,nta_arr,_ns = tst._test_data_spk___allcases(L,M)
        spk=spk*0+1
    elif typ==3:
        _nta_arr = np.array([6*10,7*10,8*10])
        spk,_L,nta_arr,_ns = tst._test_data_spk_rand(L=L, nta_arr=_nta_arr, M=M)
    #todo: more tests*
    else:
        assert False
    return spk,nta_arr,_ns

def spk_singleL(spk,nta,M):
    #not used, not tested.
    ns = spk.shape[3]
    L = spk.shape[1]
    ntr_total=spk.shape[2]
    pwa= M**np.arange(0, L)
    assert spk.shape[0]==1
    spk1 = np.zeros((1,1,ntr_total,ns), int)
    for t in range(0,ns):
        for tri in range(0,nta[t]):
            spk1[0,0,tri,t] = spk[0,:,tri,t].flatten().dot( pwa )
        #print range(nta[t],ntr_total)
        #AA=spk[:,nta[t]:ntr_total,t]
        #BB=spk1[:,nta[t]:ntr_total,t]
        assert sum(abs(spk[:,nta[t]:ntr_total,t]) .flatten() )==0  #flatten is necessary
        assert sum(abs(spk1[:,nta[t]:ntr_total,t]) .flatten() )==0


import sys
def print_flat(X):
    for i in range(0,len(X)):
        sys.stdout.write(str(X[i])+" ")
    sys.stdout.write('\n')


class Tests_probs(unittest.TestCase):
  def test_1(self):
        M=3
        for L in [5,2,1]:
            for typ in [1,2,3]:
                spk,nta_arr,ns=generate_spka(L,M, typ)
                for q in filter(lambda q: q<=L, [0,1,3,L]):
                        #print typ,spk.shape
                        #ns=spk.shape[3]
                        #prs_whole=np.zeros((M**L,ns),float)
                        #print "nta_arr",nta_arr, "L=",L, "M=",M,"typ",typ
                        for si in range(0,ns):
                            #_range = range_shuffle(nta_arr)
                            #r43 = range_frac(_range, nta_arr, f, k)
                            def randperm(n):
                                return np.random.permutation(range(n))+1

                            #idx = randperm(nta_arr[si])
                            #r43 = idx[:]

                            spkq = q_shuffle(spk, nta_arr, q)

                            ns_q = spkq.shape[3]
                            L_q = spk.shape[1]
                            #spksq_sorted = np.zeros(spkq.shape, int)

                            for t in range(0, ns_q):
                                #print "t=",t
                                #spksq_sorted = spksq_sorted[0,L];
                                #A = spksq_sorted[0,:,:,t]
                                #B = spk[0,:,:,t]
                                A = spk[0,:,:,t]
                                B = spkq[0,:,:,t]
                                A=A.flatten()
                                B=B.flatten()
                                A.sort()
                                B.sort()
                                #print 'A=',A.T
                                #print 'B=',B.T
                                #print 'A=',
                                #print_flat(A)
                                #print 'B=',
                                #print_flat(B)
                                #print nta_arr,t
                                assert sum( abs(A.flatten() - B.flatten() ) ) == 0
                                #a weak test
                                #todo: also test if they are preserved up to q

                            #assert sum( abs(spkq_sorted.flatten() - spk.flatten() ) ) == 0
                            notused=spk_singleL(spkq,nta_arr,M)


if __name__ == '__main__':
    unittest.main()


"""
function [spks]=q_shuffle(spk,nt,q)
%does a shuffling preserving marginals up to order q
ns=size(spk,4);
L=size(spk,2);
spks1=zeros(size(spk));
for t=1:ns
    %over all stimulus conditions
    rbin=randperm(nt(t));
    M=1+max(reshape(spk(1,:,:,t),1,[]));
    spks(1,1,1:nt(t),t)=spk(1,1,rbin,t); %first bin comes from random picking of first bins
    for k=2:L   %now we have to choose one by one the next bins
        if (q>k-1)
            qe=k-1;
        else
            qe=q;
        end
        memory=squeeze(spk(1,k-qe:k-1,1:nt(t),t))'; %past spikes
        memorys=squeeze(spks(1,k-qe:k-1,1:nt(t),t))';
        if (qe==1)
            memory=memory';
            memorys=memorys';
        end
        memd=memory*[M.^[0:qe-1]]';
        memds=memorys*[M.^[0:qe-1]]';
        for g=0:M^qe-1
            Igs=find(memds==g);
            Ig=find(memd==g);
            new_spike=spk(1,k,Ig,t);
            %mix them
            new_spike=new_spike(randperm(length(Ig)));
            %put them back
            spks(1,k,Igs,t)=new_spike;
        end
    end
%end
%return spks
"""
