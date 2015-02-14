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
