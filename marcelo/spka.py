#Experimental
import hx
import numpy as np

class Spk1LNS():
    def __init__(self, nta_arr,L=None, generator=None,spka=None):
        #M
        self.nta=np.array(nta_arr,hx.COUNT_TYPE)
        ns=len(self.nta)
        hx._checktype_nta(self.nta, ns)
        if spka==None:
            self.spk = np.zeros([1,L,max(self.nta),ns], hx.RESP_TYPE)
            for i in range(1):
                for l in range(L):
                    for s in range(ns):
                        for tr in range(self.nta[s]):
                            r = generator(s,tr,l,self.nta)
                            #assert type(r) is hx.RESP_TYPE
                            self.spk[i,l,tr,s] = hx.RESP_TYPE(r)
        else:
            assert generator is None
            assert L is None
            self.spk = spka

        hx._checktype_spk1LNS(self.spk, self.nta)
        #return spk,L,nta,ns

    def get_L(self):
        return self.spk.shape[1]
    def get_ntr(self):
        """total number of trials"""
        return sum(self.nta)
    def get_ns(self):
        return self.spk.shape[3]

    def __repr__(self):
        ns = self.get_ns()
        L =  self.get_L()
        r=str("")
        #r+= "%dxL=%dxN=%dxs=%d\n"%spk.spk.shape
        r+= "1xLxNxS=%dx%dx%dx%d\n"%spk.spk.shape
        for i in range(1):
           for s in range(ns):
              r+="%d: "%(s,)
              for tr in range(self.nta[s]):
                 for l in range(L):
                     r = r + str(int(self.spk[i,l,tr,s]))
                 r+=" "
              r+="\n"
        return r[0:len(r)-1]

    #def counts():
    #    pass
    #def probrs():
    #    pass

    def hr(self, biasType):
        return hx.hr(self.spk,self.nta,biasType)

    def hrs(self, biasType):
        return hx.hrs(self.spk,self.nta,biasType)

    #def mi(biasType):
    #    return self.hr(biasType) - self.hrs(biasType)

#class Counts():
#    def __init__():
#        pass
#    pass
#
#class probs():
#    def __init__(counts):
#        assert type(counts) is Counts
#        pass

#generator(s,tr,l,self.nta)

#nta_arr=[100,100]
#L=2
#spk = spk1LNS( lambda s,tr,l,nta: s , nta_arr,L)

def round001(x):
    FAC=1000000.0
    return round(x*FAC)/FAC


@hx.test_tdd
def test_spk1LNS():
    spk = Spk1LNS(nta_arr=[15,13], L=2, generator=lambda s,tr,l,nta: s)
    print spk.spk.shape
    print repr(spk)
    print spk.nta
    (a,b)=(spk.hr(1), spk.hrs(1))
    print round001(a-b)

    NTA = [1000,1000,1000]
    spk1,L,nta,ns = hx._test_data_spk_rand(L=2,nta_arr=NTA,M=5)
    spk = Spk1LNS(nta_arr=nta, spka=spk1)
    print repr(spk)
    print round001(spk.hr(1) - spk.hrs(1))

    spk = Spk1LNS(nta_arr=[150,130], L=2, generator=lambda s,tr,l,nta: s+2*np.random.randint(0,2))
    print spk.spk.shape
    print repr(spk)
    print spk.nta
    (a,b)=(spk.hr(1), spk.hrs(1))
    print round001(a-b)

def sliding(zq_TxN, L, step=1, offset=0, downsample=False): #also step between L elements?
    """ converts a TxN into LxSN : To convert continuous signals into spka format """
    #assert type(zq_TxN[0,0]) is int
    #zq: (nlen,ntr) = (T,N)
    #assert step==1
    import numpy,math
    ns = int(math.floor((zq_TxN.shape[0]-L+1)/step)) #only tested for step==1
    #print "ns=%d"%ns
    ntr=zq_TxN.shape[1]
    #zL=numpy.zeros((ns,ntr), type(zq[0,0]))
    if step>1:
        print "Warning: todo: down-sample"
        #todo: downsample types
        if downsample:
            raise Exception("Not implemented")

    z_LxSN=numpy.zeros((L,ns*ntr), type(zq_TxN[0,0]))
    #nta = numpy.zeros(nlen, type(zq[0,0]))
    nta = numpy.zeros(ns,int)
    print type(nta[0])
    word_start = offset #argument: offset
    SN_ctr = 0 #The SN dimension is (Stim x Ntr), i.e, arrays of ntr repeated Stim (=ns) times.  Lx SN=Lx(Stim x Ntr)
    for i in range(ns):
        a=zq_TxN[0+word_start:L+word_start,:]  # Lx30
        z_LxSN[0:L,SN_ctr+0:SN_ctr+ntr] = a
        #start = start + 1   #todo: start=i
        word_start = word_start + step  #todo: start=i*step
        nta[i] = ntr
        SN_ctr=SN_ctr+ntr #todo: ntrctr = i*ntr
    return z_LxSN,nta

def Spka_from_signal_TxN(zq_TxN,L):
    #not tested
    z_LxSN,nta = sliding(zq_TxN, L, step=1, offset=0, downsample=False)
    spk1 = z_LxSN.reshape([1]+z_LxSN.shape)
    return Spk1LNS(nta_arr=nta, spka=spk1)

def quantize_TxN(z_TxN, M, uniform_code):
    raise Exception("Not implemented")
    z_NxT=z_TxN.transpose()
    z_NxT_shape=z_NxT.shape
    z_Arr=z_NxT.reshape([z_NxT.size]) #straighted.  z2d.size=total elements
    zq_Arr,bin_bounds, bin_centers = pyentropy.quantise(z_Arr, M, uniform=uniform_code)
    zq_NxT = zq_Arr.reshape(z_NxT_shape)
    return zq_NxT.transpose()



class Quantizer:
    def quantize(s):
        #assert len(s.shape)==1
        assert type(s) is np.array
        raise "abstact class"


#QUANTIZERS=Quantizer.__subclasses__()

class Signal_TxN:
    """ For keeping responses in te format of time_samples x trials. This is very useful for continuous signals."""
    #quantize_TxN
    def __init__(self, signal_TxN, ntr=-1):
        """ ntr
        :param signal_TxN: signal time_samples x trials. Can be either array of float or int.
        :param ntr: is mandatory
        :type ntr: int
        """
        assert len(signal_TxN.shape)==2
        self.signal_TxN = signal_TxN
        if ntr == -1:
            raise Exception("ntr argument is mandatory.")
        assert self.signal_TxN.shape[1]==ntr
        #self.state='raw'
    def _state_is_continuous(self):
        return type(self.signal_TxN[0,0]) in [int, np.int8,np.int16,np.int32,np.int64] #wrong
    def _state_is_discrete(self):
        return not self._state_is_continuous()
        #return not type self.signal_TxN[0,0] in [float, np.float] #wrong
    def resample(self,N, method):
        assert self._state_is_continuous()
        raise Exception("Not implemented")
    def quantise(self,M,method):
        assert method in Quantizer.__subclasses__(), "Quantizer can be one of the following: [%r]"%(Quantizer.__subclasses__(),)
        self.signal_TxN = quantize_TxN(self.signal_TxN, M, method)
        #self.state='quantised'
    def get_Spka(self):
        #assert self.state=='quantised'
        assert self._state_is_discrete()
        nta=[self.signal_TxN.shape[1]]
        return Spka_from_signal_TxN(self.signal_TxN, nta)
