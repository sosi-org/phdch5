
import numpy as np
import scipy
import matcompat
import matplotlib.pylab as plt
from consts import EPS, BIG_EPS

"""
import string
digs = string.digits + string.letters
def int2base(x, base):
  if x < 0: sign = -1
  elif x == 0: return digs[0]
  else: sign = 1
  x *= sign
  digits = []
  while x:
    digits.append(digs[x % base])
    x /= base
  if sign < 0:
    digits.append('-')
  digits.reverse()
  return ''.join(digits)
"""
def int2base_digits(x, base):
    assert  x >= 0
    if x==0:
        return [0]
    digits = []
    while x:
       digits.append(x % base)
       x /= base
    #digits.reverse()
    return digits

i2bd = int2base_digits

def test_i2d():
    assert i2bd(0,2)==[0]
    assert i2bd(1,2)==[1]
    assert i2bd(2,2)==[0,1]
    assert i2bd(8,2)==[0,0,0,1]
    assert i2bd(7,2)==[1,1,1]
    assert i2bd(752,10)==[2,5,7]
    assert i2bd(9,3)==[0,0,1]

test_i2d()

#def dec2BaseArr(a,b):
#    assert 
#    pass

def probrind(spk, nt, range_, f):

    # Local Variables: twi, spk, swi, pt, lm, ns, nt, prob, count, new_nt, M, L, wi, wi1, pbin, Nmax, spikes, f, i, j, n, p, range, t
    # Function calls: str2num, prod, floor, max, sum, eps, diag, dec2base, length, zeros, reshape, probrind, histc, squeeze, probr, size
    #%this computes  P_ind(r)
    #%k indicates the fraction of trials to consider
    #%ntr=size(spk,3);

    new_nt = nt/f #np.floor((nt/ f))
    ns = matcompat.size(spk, 4.)
    L = matcompat.size(spk, 2.)
    #M = matcompat.max(np.reshape(spk, 1., np.array([])))+1.
    M = max(spk.flatten())+1
    if M == 1.:
        M = 2.
    
    
    p = np.zeros((M**L,))
    #p = np.zeros([1., (M** L)])
    if L > 1:
        #twi=[0:M^L-1]'; #'     M^Lx1
        #swi=dec2base(twi,M);   M^L x L   :  000, 001, ...
        #wi=zeros(M^L,L);
        #for i=1:L
        #    wi(:,L-i+1)=str2num(swi(:,i));  % str2num(swi(:,i)) means digit i (L-i) of all
        #end

        #twi = np.array(np.hstack((np.arange(0., (matixpower(M, L)-1.)+1)))).conj().T
        #twi = np.arange(0, M**L-1+1).T
        
        #swi = dec2base(twi, M)
        #wi = np.zeros([M** L, L])
        #for i in np.arange(1., (L)+1):
        #    wi[:,int((L-i+1.))-1] = str2num(swi[:,int(i)-1])
        #    #put digit i of number j into wi[j,L-(i-1)]
        
        twi = np.arange(0, M**L)
        wi = np.zeros([M** L, L], int)
        #assert type(M) is int

        for j in range( len(twi) ):
            v = int(twi[j])
            for i in range(0, L):
                wi[j,L-1-i] = v % M #digit(i of twi[j])
                v=int(v/M)
            assert v==0
        
        
        #%ntrk=size(range_,2); 
        for t in range(1, (ns)+1):
            spikes = np.squeeze(spk[0,:,:,int(t)-1]).conj().T
            #%trials set for current stimulus condition%now get the probabilities for each bin
            #        spikes=spikes(range_(t,1:new_nt(t)),:);
            assert int(t)-1 >=0
            assert int(t)-1 < len(range_)
            assert new_nt[int(t)-1] > 0
            assert spikes.shape[1] > 0
            #print new_nt[:5],"O"
            #print spikes.shape[0] , new_nt[int(t)-1], t
            assert spikes.shape[0] >= new_nt[int(t)-1]
            #spikes = spikes[range_[int(t-1),0:new_nt[int(t)-1]],:]
            rr = range_[int(t-1),0:new_nt[int(t)-1]]
            #print min(rr),max(rr) #1,16
            #print spikes.shape #16x2
            spikes = spikes[rr-1,:] # -1 was missing
            prob = np.zeros([L, M])
            for i in np.arange(1., (L)+1):
                n = spikes[:,int(i)-1]
                Nmax = M-1.
                pbin = np.zeros([1, (Nmax+1)])
                count = np.zeros([(Nmax+1.), 1])
                wi1 = 1.+n
                #count = histc(wi1, np.array(np.hstack((np.arange(1., (Nmax+1.+EPS)+1)))))


                maxrange = 1+(BIG_EPS+Nmax)+1
                edges = np.array(range(1,int(maxrange)))+0.1 #EPS does not work here!! #bin edges, including the rightmost edge,
                #count,e2 = np.histogram(wi1.flatten(), edges) #
                #assert sum(count) == len(wi1.flatten())
                count,e2 = np.histogram(wi1, edges) #
                #print wi1.shape #(16,)
                #print count
                assert np.sum(count) == wi1.shape[0]



                pbin = count / float(np.sum(count))
                #print pbin.shape, "pbin*"
                #print edges, Nmax [ four numbers], 3
                lm = len(pbin)  #size=(3,)
                #print lm
                #print i
                #print pbin
                #print prob.shape #2,
                prob[int(i)-1,0:lm] = pbin
                
            #print matcompat.size(p)  # 1x16
            #pt = np.zeros(matcompat.size(p))
            pt = np.zeros((M**L,))
            #%size(wi),t,M
            for j in np.arange(1, (M ** L)+1):
                #            pt(j)=prod(diag(prob([1:L],1+wi(j,:))));
                #pt[int(j)-1] = np.prod(np.diag(prob[np.array(np.hstack((0:L))),(1.+wi[int(j)-1,:])]))
                #print "**",
                #print j-1,
                #print pt.shape
                #print prob.shape, wi[int(j)-1,:], L #L=2, prob: 2x4, wi(,:)==[0,3] j==4,
                pt[int(j)-1] = np.prod(np.diag(prob[np.arange(0,L,dtype=int),1-1+wi[int(j)-1,:]]))
                #latest error:
                #IndexError: index 1 is out of bounds for axis 0 with size 1
                #maybe p does not have to be 1x..., but is  should be one-dimenisional. but what about .dot (below) ?

                
            p = p+np.dot(pt, new_nt[int(t)-1])
            
        p = p / float( np.sum(new_nt))
    else:
        p = probr(spk, nt, range_, f)
        
    
    return p