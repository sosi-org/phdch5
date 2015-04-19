import numpy as np
import scipy
import matcompat

#import matplotlib.pylab as plt
import myhist

#checklist:  M, check.
#histc: done

def pqmargs(spikes, L, q):
    """ q: markov order
    """
    assert type(q) is int
    #print "L=%d, q=%d"%(L,q)
    #assert spikes.shape==(trials,L)
    (debug_trials,debug_L) = spikes.shape #just for annotation. Invar-time variables. Invar-time can be rutime or dbug or compile-time. They may move between times.  If this variable is used later, it's for consistency chekck. The main function is anotation (naming). Like: ##"The number of trials" := x.shape[0]  #PLANGNOTE

    assert L==debug_L, "Not sure about this test"

    # Local Variables: spikes, words, nn, i, npd, M, L, wi, q, npn, ws, il, pd, pn, nn2
    # Function calls: max, sum, eps, pqmargs, zeros, reshape, histc
    #M = 1.+matcompat.max(np.reshape(spikes, 1., np.array([])))
    #print spikes.shape
    assert spikes.shape[1] > 0
    assert spikes.shape[0] > 0

    M = 1+np.max(spikes.flatten())
    if M == 1:
        #print "M"
        M = 2
    
    #%Pr(r(l)|r(l-1),r(l-2),...r(l-q),s) ? %yes. q+1 becasue of the one on the left     
    #pn = np.zeros([matixpower(M, q+1.), L], int)     
    pn = np.zeros([(M** (q+1)), L])     
    #% Pr(r(l),r(l-1),r(l-2),...r(l-q)|s) ?!     
    pd = np.zeros([(M ** (q+1)), L])     
    #% Pr(r(l-1),r(l-2)...r(l-q)|s) % |s ??     
    #%ntrials x L  (l=1)   r_l ?     
    ws = 1     
    words =spikes[:,0].reshape((debug_trials,ws))     
    #print words.shape
    assert words.shape == (debug_trials,ws)
    #wi = 1.+np.dot(words,(M**np.array(np.hstack((np.arange(0., (ws-1.)+1))))).T)     
    M_pw_ws =(M**np.arange(0, ws-1+1)).reshape((ws,1))
    wi = 1+words.dot(M_pw_ws)

    #%% indices (numbers to base M)     
    #pn[0:M,0] = pn[0:M,0]+histc(wi,np.array(np.hstack((np.arange(1., (matixpower(M, ws)+EPS)+1)))))     
    assert wi.shape==(debug_trials,ws)
    nn0 = M ** ws     
    #print pn.shape
    xx=myhist.myhist_int(wi, nn0, startWith1=True)
    #print xx.shape
    pn[0:M,0] = pn[0:M,0]+myhist.myhist_int(wi, nn0, startWith1=True)    
    #histc(wi,np.array(np.hstack((np.arange(1., ((M ** ws)+EPS)+1)))))     
    #%count each counter bin

    pd[:,0] = np.nan

    #for i=2:L
    #    il=i-q;
    #    if il<1
    #        il=1;
    #    end
    #    ws=i-il+1;
    #    nn=M^ws;
    #    %%%%%%%%%%%%%%%%%%%%%%%
    #    words=spikes(:,il:i);
    #    wi=1+words*(M.^[0:ws-1])';
    #    pn(1:nn,i)=pn(1:nn,i)+histc(wi,[1:nn+eps]);
    #    %%%%%%%%%%%%%%%%%%%%%%%
    #    ws=i-il;
    #    nn2=M^ws;
    #    words=spikes(:,il:i-1);
    #    wi=1+words*(M.^[0:ws-1])';
    #    pd(1:nn2,i)=pd(1:nn2,i)+histc(wi,[1:nn2+eps]);
    #end

    #a[0,0].shape  --> Out[6]: ()

    for i in range(2,L+1):   #np.arange(2., (L)+1):
        il = i-q
        if il<1:
            il = 1
        #  i = q+(il)
        #  i = (ws-1) + il
        #  i = ws + (il-1)
        # 1,2,3,...,il,...,i.
        #           --------.  words(1)
        #           ------,    words(2)
        ws = i-il+1

        #i-il+1>=i-(i-q)+1=q+1
        assert ws<=q+1
        #print "ws",ws
        #nn = matixpower(M, ws)
        nn = M ** ws
        #%%%%%%%%%%%%%%%%%%%%%%%
        words = spikes[:,int(il-1):(i+1-1)]  #trials x (<q)
        #print "words.shape",words.shape
        #print words.shape , (debug_trials,q)  # (10,3) (10,2)
        #print ws,q,words.shape[1] #3,2,3
        assert words.shape[1] >= q
        assert words.shape >= (debug_trials,q)
        #print wi.shape, (debug_trials,ws), q
        #print M_pw_ws
        #print wi.T
        #(10, 1) (10, 2)
        assert words.shape == (debug_trials,ws)
        #wi = 1.+np.dot(words, (M**np.array(np.hstack((np.arange(0., (ws-1.)+1))))).conj().T)
        M_pw_ws = (M**np.arange(0, ws-1+1)).reshape((ws,1)) #reuse (at what level/loop?) #PLANGNOTE. CONST==compile-time-reuse
        #wi = 1+np.dot(words, (M**np.arange(0, ws-1+1)))
        #wi = 1+np.dot(words, M_pw_ws)
        wi = 1+words.dot(M_pw_ws)
        #print wi.shape, ws
        #assert wi.shape==(debug_trials,1)
        #print wi.shape, (debug_trials,1) # #PLANGNOTE: debug_time or invar_time?
        assert wi.shape==(debug_trials,1)
        #print words.shape, (debug_trials,ws)
        assert words.shape==(debug_trials,ws)

        assert type(i) is int
        mh = myhist.myhist_int(wi, nn, startWith1=True)
        #print mh.shape,"**"
        #pn[0:nn,int(i)-1] = pn[0:nn,int(i)-1]+histc(wi, np.array(np.hstack((np.arange(1., (nn+EPS)+1)))))
        #pn[0:nn,int(i)-1] = pn[0:nn,int(i)-1]+myhist.myhistint(wi-1, nn+1)
        pn[0:nn,int(i)-1] = pn[0:nn,i-1] + myhist.myhist_int(wi, nn, startWith1=True) #tie to an assert as a requirement. #PLANGNOTE  #righteous programming paradigm.
        #%%%%%%%%%%%%%%%%%%%%%%%
        ws = i-il
        #    nn2=M^ws;
        nn2 = M ** ws
        assert type(il) is int
        #    words=spikes(:,il:i-1);
        words = spikes[:,int(il-1):(i-1+1-1)]
        #print words.shape, "pd .shape"
        #wi = 1.+np.dot(words, (M**np.array(np.hstack((np.arange(0., (ws-1.)+1))))).conj().T)
        M_pw_ws = (M**np.arange(0, ws-1+1)).reshape((ws,1))
        #    wi=1+words*(M.^[0:ws-1])';
        wi = 1+words.dot(M_pw_ws)
        #print wi.shape, (debug_trials,ws)
        #assert wi.shape==(debug_trials,ws)
        #print wi.shape, (debug_trials,1)
        assert wi.shape==(debug_trials,1)
        #print words.shape, (debug_trials,ws)
        assert words.shape==(debug_trials,ws)

        #pd[0:nn2,int(i)-1] = pd[0:nn2,int(i)-1]+histc(wi, np.array(np.hstack((np.arange(1., (nn2+EPS)+1)))))
        pd[0:nn2,int(i)-1] = pd[0:nn2,i-1] + myhist.myhist_int(wi, nn2, startWith1=True)
        #    pd(1:nn2,i)=pd(1:nn2,i)+histc(wi,[1:nn2+eps]);

        #assert pn.shape==(M**(q+1),L)

    #size(sum(zeros(3,5)))  ---> (1,5)
    #sum(np.zeros([3,5]))).shape  ---> (5,)
    #print pn.shape,  # 8x5
    #print pd.shape   # 8x5
    #print np.sum(pn).shape,np.sum(pd).shape, "*"# () !!!
    #print sum(pn).shape,sum(pd).shape, "**"   # (5,) !!!

    npn = sum(pn)
    npd = sum(pd)
    #print npn.shape,  # (5,)
    #print npd.shape   # (5,)


    #pn[:,0] = matdiv(pn[:,0], npn[0])
    #print npn #  10,10,10,...
    #print npd #   0,10,10,...
    #print npn[0] # is 10
    #print npd[0] #is zero

    #print pn[:,0].shape

    pn[:,0] = pn[:,0] /  npn[0]
    #for i in range(2, L+1):
    #    pn[:,int(i-1)] = pn[:,int(i)-1] / npn[int(i)-1]
    #    pd[:,int(i-1)] = pd[:,int(i)-1] / npd[int(i)-1]
    for j in range(1, L):
        pn[:,j] = pn[:,j] / npn[j]
        pd[:,j] = pd[:,j] / npd[j]

    #pn = Pr(r|?)

    pn = pn.T
    pd = pd.T
    return pn, pd


#%Pr(r(l)|r(l-1),r(l-2),...r(l-q),s) ? %yes. q+1 becasue of the one on the left
#pn=zeros(M^(q+1),L);  % Pr(r(l),r(l-1),r(l-2),...r(l-q)|s) ?!
#pd=zeros(M^(q+1),L);  % Pr(r(l-1),r(l-2)...r(l-q)|s) % |s ??

#pn[v,l] = Pr(v,?|:)
#q=0 ==> pd[:] = [1,0,0,0,0]. why? because there is no v!  It is basically Pr(0|...) It will not be used.

import hx_test_utils as tst
import unittest

class TestM(unittest.TestCase):


  def test_pqmargs(self):
    for q in [2,1,0]:
        for L in [5,2,1]:
            nt=10
            spikes=np.zeros([nt,L])
            #print L,q,"L,q"
            pn, pd = pqmargs(spikes, L, q)
            #print pn
            #print pd
            self.assertEqual('foo'.upper(), 'FOO')

            spikes=np.zeros([nt,L])+1
            spikes[0,0]=0
            pn, pd = pqmargs(spikes, L, q)
            #print pn
            #print pd

            M=3
            nt=M**L
            spikes=tst.n_ary(np.arange(0,M**L),L,M)
            assert spikes.shape == (nt,L)
            pn, pd = pqmargs(spikes, L, q)
            #print pn
            #print pd

  """
  def test_1(self):
        for L in [5,2,1]:
            for typ in [1,2,3]:
                if typ==1:
                    spk,_L,nta_arr,_ns = tst._test_data_spkNtL___allcases(L=L,M=3)
                elif typ==2:
                    spk,_L,nta_arr,_ns = tst._test_data_spkNtL___allcases(L=L,M=3)
                    spk=spk*0+1
                elif typ==3:
                    _nta_arr = np.array([6*10,7*10,8*10])
                    spk,_L,nta_arr,_ns = tst._test_data_spkNtL_rand(L=L, nta_arr=_nta_arr, M=3)
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
  """

  #def test_isupper(self):
  #    self.assertTrue('FOO'.isupper())
  #    self.assertFalse('Foo'.isupper())##
  #
  #def test_split(self):
  #    s = 'hello world'
  #    self.assertEqual(s.split(), ['hello', 'world'])
  #    # check that s.split fails when the separator is not a string
  #    with self.assertRaises(TypeError):
  #        s.split(2)

if __name__ == '__main__':
    unittest.main()