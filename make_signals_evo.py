#import myshared
import exrxp
#import numpy
    
    
#main
ntr=10
#nlen=100000/100
nlen=1000

#ntr=30
#ntr=5
#nlen=10

fs_Hz=1000.0 # Hz

tau_msec = 1.0 #10 msec
z = exrxp.exrxp (nlen,tau_msec/1000.0,fs_Hz) 
z2d = exrxp.exrxp_ntr (nlen,tau_msec/1000.0,fs_Hz, ntr)  

ts = exrxp.timesarr(nlen,fs_Hz)

if 0:
    import matplotlib.pyplot
    p1=matplotlib.pyplot.plot(ts,z)
    p1=matplotlib.pyplot.plot(ts,z2d)
    matplotlib.pyplot.show()


import pyentropy


#M = 4
M = 10
if 0:
    #import pyentropy
    #z_q = pyentropy.quantise(z, M, uniform='bins') #or 'sampling'
    #z_q,a,b = pyentropy.quantise(z, M, uniform='bins') #or 'sampling'
    #return q_value, bin_bounds, bin_centers
    #why would he need bin_centers?
    z_q,bin_bounds, bin_centers = pyentropy.quantise(z, M,
                                                     uniform='sampling') 
                                                     #uniform='bins') #or 'sampling'
                                                     
    print len(z_q)

    #print ("%r %r %r    "%(z_q,a,b)) #z_q.shape
    #print ("%r "%(z_q)) 
    #print ("%r "%a)
    #print ("%r "%b)
    from pyentropy import DiscreteSystem
    s = DiscreteSystem(z_q,(1,M), z_q,(1,M))
    #print("***")
    #Warning: Null output conditional ensemble for output : 4 (when M is larger than previous M)
    #s.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
    s.calculate_entropies(method='qe', calc=['HX', 'HXY'])
    #print help(s)  #Ish, Ispike, Ishush, ...
    print s.I()
    #1.73547281807



#git clone git@github.com:robince/pyentropy.git
#sudo get-apt install libgsl0-dev
#sudo apt-get  install libgsl0-dev
#sudo get-apt install build-essential python-dev libgsl0-dev
#sudo apt-get  install build-essential python-dev libgsl0-dev
#matlab is sad. matlab is not a software. It's a zone.
#... test()

#PyEntropy: 
#If you're input data is continuous, it must first be quantised using pyentropy.quantise() to a suitable discrete representation


#import pyentropy
#pyentropy.test()
#import numpy as np
#from pyentropy import DiscreteSystem
#x = np.random.random_integers(0,9,10000)
#y = x.copy()
#indx = np.random.permutation(len(x))[:len(x)/2]
#y[indx] = np.random.random_integers(0,9,len(x)/2)
#s = DiscreteSystem(x,(1,10), y,(1,10))
#s.calculate_entropies(method='plugin', calc=['HX', 'HXY'])
#s.I()
#s.calculate_entropies(method='pt', calc=['HX', 'HXY'])
#s.I()
#s.calculate_entropies(method='qe', calc=['HX', 'HXY'])
#s.I()
#s.calculate_entropies(method='nsb', calc=['HX', 'HXY'])  #did not work
#np.log2(10) - (-9*0.05*np.log2(0.05) - 0.55*np.log2(0.55))



if 0:
    import matplotlib.pyplot
    matplotlib.pyplot.plot(ts,z2d)
    matplotlib.pyplot.show()


def quantize_2d(z2d, M, uniform_code):
    #z2d_reshaped=z2d.reshape(z2d.size)
    #z2d_reshaped_q,bin_bounds, bin_centers = pyentropy.quantise(z2d_reshaped, M, uniform='bins')
    #z2d_q = z2d_reshaped_q.reshape(z2d.shape)
    z2d_reshaped=z2d.reshape(z2d.size)
    z2d_reshaped_q,bin_bounds, bin_centers = pyentropy.quantise(z2d_reshaped, M, uniform=uniform_code)
    z2d_q = z2d_reshaped_q.reshape(z2d.shape)
    return z2d_q

def sliding(zq, L, step=1): #also step between L elements?
    #zq: (nlen,ntr)
    assert step==1
    import numpy
    ns = int(numpy.floor((zq.shape[0]-L+1)/step)) #only tested for step==1
    #print "ns=%d"%ns
    ntr=zq.shape[1]
    #zL=numpy.zeros((ns,ntr), type(zq[0,0]))
    zL=numpy.zeros((L,ns*ntr), type(zq[0,0]))
    #nta = numpy.zeros(nlen, type(zq[0,0]))
    nta = numpy.zeros(ns,int)
    #nl = zq[
    start = 0
    ntrctr=0
    for i in range(ns):
        #print start
        a=zq[0+start:L+start,:]  # Lx30
        #print a.shape #2x30
        #print (0+start,L+start) #0,2 for L=2
        #zL[i,:]=zq[0+start:L+start,:]
        #print zL[i+0:i+L,:]
        #zL[i+0:i+L,:]=zq[0+start:L+start,:]
        #zL[i+0:i+L,:] = a
        zL[0:L,ntrctr+0:ntrctr+ntr] = a
        #start = start + (step-1+L)
        start = start + 1
        nta[i] = ntr
        ntrctr=ntrctr+ntr
    return zL,nta


z2d = exrxp.exrxp_ntr (nlen,tau_msec/1000.0,fs_Hz, ntr)

z2d_q = quantize_2d(z2d, M, 'sampling') #'bins')
#print z2d_q.shape
#z2dqL,nta = sliding(z2d_q, L=1)
z2dqL,nta = sliding(z2d_q, L=2)
#z2dqL,nta = sliding(z2d, L=2)

#print z2dqL.shape 2x9990 

#print "---"

if 0:
    import matplotlib.pyplot
    print "***"
    print range(z2dqL.shape[0]) #[0,1]
    print z2dqL[0,:].shape #[270] ----> 9 x 30
    #there is an error ad its ok
    matplotlib.pyplot.plot(range(z2dqL.shape[0]),z2dqL[0,:])
    matplotlib.pyplot.show()

if 0:
    print z2d_q.shape
    print z2d.shape
    print z2dqL.shape

    #print z2dqL.T
    #print z2dqL[:,0:20].T
    #print z2d_q[:,0:10].T
    print z2dqL.T
    print z2d_q.T


#exit(0)


#SortedDiscreteSystem(X, X_dims, Ym, Ny)
#s = DiscreteSystem(z2d_q,(1,M), z2d_q,(1,M))
import numpy
#nta=numpy.zeros(nlen)+ntr
#print nta
#s= SortedDiscreteSystem(z2d_q, (1,M), M, nta)
from pyentropy import SortedDiscreteSystem
#s= SortedDiscreteSystem(z2d_q, (1,M), nlen, nta) 
#z2dq.shape[0] must be eq Xn
#print nta.sum()
print z2dqL.shape
#s= SortedDiscreteSystem(z2dqL, (z2d_q.shape[0],M), nlen, nta)
#s.calculate_entropies(method='qe', calc=['HX', 'HXY'])
#print s.I()

s = SortedDiscreteSystem(z2dqL, (z2dqL.shape[0],M), len(nta), nta)
s.calculate_entropies(method='qe', calc=['HX', 'HXY'])
print s.I()

#SortedSystem:(X, X_dims,Y_m,Ny)
#
#Check and assign inputs.
#
#:Parameters:
#  X : (X_n, t) int array
#    Array of measured input values. X_n variables in X space, t trials
#  X_dims : tuple (n,m)
#    Dimension of X (input) space; length n, base m words
#  Y_m : int 
#    Finite alphabet size of single variable Y
#  Ny : (Y_m,) int array
#    Array of number of trials available for each stimulus. This should
#    be ordered the same as the order of X w.r.t. stimuli. 
#    Y_t.sum() = X.shape[1]
#
#
#

#z2d_q: (1000,10) = (nlen,ntr)
# X_n = nlen = 1000
# t = 10 = ntr
# X_dims = (?,M)
# Y_m=len(?)




# for i in xrange(self.Y_dim):
#               send = sstart+self.Ny[i]
#               indx = slice(sstart,send)
#               sstart = send
#
