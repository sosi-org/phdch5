import numpy
import numpy.random
import numpy as np
import myshared #verbose

def exrxp ( n , tau_sec , fs_Hz, KLENRATIO=5 ):
    """
     @type n: int
     @type tau_sec: float
     @type fs_Hz: float
    """
    tau_sec=float(tau_sec)
    fs_Hz=float(fs_Hz)
    
    # the kernel function :
    klen = KLENRATIO * tau_sec * fs_Hz
    assert tau_sec>0 #don't include 0
    
    klen=int(numpy.ceil(klen))
    #or insted of ceil, do this:
    assert klen>0
    #not sure
    if klen<1:
        verbose("Warning: kernel len is small: %d. making it=1"%klen)
        klen=1

    #assert klen<n*100

    assert klen<n*1000
    assert n>0
    assert n*klen<1E7 #takes up to a few seconds. todo: improve these conditions.
    #klen is ready

    #t = [ 0 : ( klen -1) ] / fs_Hz ;
    t = np.linspace(0 , klen -1, klen) / fs_Hz
    k = np.exp ( -t / tau_sec )
    x = np.random.randn (n + klen *2 + 4 )     # the white noise :
    #lim = [ klen +3 , n + klen +2] #matlab
    lim = [ klen +3 , n + klen +3] #the size is different to matlab
    # 'full' is the default mode. 
    r = np.convolve ( x , k / fs_Hz , 'full') * fs_Hz 
    # fixing the phase - shift :
    r = r [ lim [0] : lim [1] ]
    #for test: r=numpy.cumsum(r)
    return r

def exrxp_ntr ( n , tau_sec , fs_Hz , ntr):
    sig = numpy.zeros((n,ntr), float)
    for i in range(ntr):
        sig[0:n,i] = exrxp(n , tau_sec , fs_Hz)
    return sig

#todo: move to myshared
def timesarr(n,fs_Hz):
    return np.linspace(0 , n-1, n) / fs_Hz
    # x = np.arange(0, 5, 0.1) #also try this

#@test
def test_exrxp():
    for nlen in [1,1000]:
        #z=erp1(10,0.1,fs_Hz)
        #z=erp1(100,0.001,fs_Hz)
        #z=erp1(100,1000,fs_Hz)
        fs_Hz=1000
        #z=erp1(l,10,fs_Hz) #10 sec is very slow.
        z=exrxp(nlen,10,fs_Hz) 
        #test: check size, check small lengths, check 0, 0.0001, ...  n=0, n=1
        assert len(z)==nlen
