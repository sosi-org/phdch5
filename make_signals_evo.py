import numpy
import numpy.random
import numpy as np

def make_lor1(ntr,nlen):
    a = numpy.random.randn(ntr,nlen)
    b= numpy.cumsum(a, axis=1 )

    if 1:
        #import plotdata
        #import matplotlib.pyplot as pyplot
        #import plotdata
        import matplotlib.pyplot
        #matplotlib.pyplot.plot(a[0,:])
        #z=matplotlib.pyplot.plot(a[0,:])
        z=matplotlib.pyplot.plot(a[0,0:500])
        z2=matplotlib.pyplot.plot(b[0,0:500])
        if 0:
            gfn='graph3.png'
            matplotlib.pyplot.savefig(gfn)
            import os
            os.system("dolphin %s" % gfn)
        #
        matplotlib.pyplot.axis([0, 500, -30, 30]) #minx maxx, miny maxy
        matplotlib.pyplot.show()
        
    #import matplotlib.pyplot as plt
    #plt.plot([1,2,3,4], [1,4,9,16], 'ro')
    #plt.axis([0, 6, 0, 20])
    #plt.show()

def make_lor1(ntr,nlen):
    a = numpy.random.randn(ntr,nlen)
    b = numpy.cumsum(a, axis=1 )


def verbose(s):
    print(s)
#np.array([1, 2, 3], dtype=complex)

def erp1 ( n , tau_sec , fs_Hz ):
    # @type n: int
    # @type tau_sec: float
    # @type fs_Hz: float
    tau_sec=float(tau_sec)
    fs_Hz=float(fs_Hz)
    
    # the kernel function :
    #klen = 5 * tau_sec / fs_Hz
    klen = 5 * tau_sec * fs_Hz
    
    assert tau_sec>0 #don't include 0
    
    klen=int(numpy.ceil(klen))
    #or insted of ceil, do this:
    assert klen>0
    #not sure
    if klen<1:
        verbose("Warning: kernel len is small: %d. making it=1"%klen)
        klen=1

    #assert klen<n*100

    verbose(klen)
    verbose(n)
    verbose(n*klen)
    assert klen<n*1000
    assert n>0
    assert n*klen<1E7 #takes up to a few seconds. todo: improve these conditions.
    #assert n*klen<1E8
    #klen is ready

        
    #t = [ 0 : ( klen -1) ] / fs_Hz ;
    t = np.linspace(0 , klen -1, klen) / fs_Hz
    k = np.exp ( -t / tau_sec )
    x = np.random.randn (n + klen *2 + 4 )     # the white noise :
    #lim = [ klen +3 , n + klen +2]
    lim = [ klen +3 , n + klen +3] #the size is different to matlab
    
    #print x.shape #114
    #print k.shape #5
    #print t.shape #5
    #print "==========="
    #print n
    # 'full' is the default mode. 
    r = np.convolve ( x , k / fs_Hz , 'full') * fs_Hz 
    # fixing the phase - shift :
    r = r [ lim [0] : lim [1] ]
    return r


def timesarr(n,fs_Hz):
    return np.linspace(0 , n-1, n) / fs_Hz
    # x = np.arange(0, 5, 0.1) #also try this
    
    
    
#main
ntr=100
nlen=100000
#make_lor1(ntr,nlen)


#z=erp1(10,0.1,fs_Hz)
#z=erp1(100,0.001,fs_Hz)
#z=erp1(100,1000,fs_Hz)
fs_Hz=1000
l=1000
#z=erp1(l,10,fs_Hz) #10 sec is very slow.
z=erp1(l,10,fs_Hz) 
#test: check size, check small lengths, check 0, 0.0001, ...  n=0, n=1
ts=timesarr(l,fs_Hz)
import matplotlib.pyplot
print "---------"
print z.shape
print ts.shape
p1=matplotlib.pyplot.plot(ts,z)
matplotlib.pyplot.show()




"""
function r = erp ( n , tau_sec , fs_Hz )
# the kernel function :
klen = 5 * tau_sec / fs_Hz ;
t = [ 0 : ( klen -1) ] / fs_Hz ;
k = exp ( -t / tau_sec ) ;
lim = [ klen +3 , n + klen +2] ;
# the white noise :
x = randn (1 , n + klen *2 + 4 ) ;
# the convolution ( integration ):
r = conv ( x , k / fs_Hz ) * fs_Hz ;
# fixing the phase - shift :
r = r ( lim (1) : lim (2) ) ;
"""






#numpy.ndarray(2)
#import numpy as np
#np.array([1, 2, 3], dtype=complex)
#np.arange(0,10,1)
## linspace(start, stop, num)
#linspace(0, 10)
#np.linspace(0, 10)
#np.linspace(0, 10,10)
#np.linspace(0, 10,11)
#np.arange(0,10,1)
#np.arange(0,11,1)
#
#history
#math.exp(np.arange(0,11,1))
#math.exp(2)
#math.exp([2,3])
#math.exp(  )
#np.linspace(0, 10,11)
#math.exp(np.linspace(0, 10,11))
#x = np.linspace(-2*np.pi, 2*np.pi, 100)
#x
#xx = x + 1j * x[:, np.newaxis]
#xx
#np.newaxis
#type(np.newaxis)
#np.exp(x)
#np.exp(xx)
#np.exp(  )
#x = np.exp(np.linspace(-2*np.pi, 2*np.pi, 100))
#np.exp(np.linspace(-2*np.pi, 2*np.pi, 100))
#history



#http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html
#http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
#http://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
