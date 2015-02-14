#!/usr/bin/python
# A port of M.A.M.'s code into Python
# External compatibility: the results and inputs are the same
# The first version is non-Pythonic

import numpy as np
import math
import sys

#VERBOSE = 1
TEST = 1
DEEP_TEST_DATA = 1 #test all elements of arrays

COUNT_TYPE = np.int8 # np.int16  #number of trials
RESP_TYPE = np.int8
PROB_TYPE = np.float32 #or 64?

COUNT_TYPE = np.int16 # np.int16  #number of trials
RESP_TYPE = np.int8 # np.int8
PROB_TYPE = np.float32 #or 64?

#np.emath.
#NOT RELIABLE: EPS = sys.float_info.epsilon  #http://stackoverflow.com/questions/23190017/is-pythons-epsilon-value-correct
#print EPS
#print sys.float_info.epsilon #2.22044604925e-16
#print PROB_TYPE(sys.float_info.epsilon)
#EPS_PROB =PROB_TYPE(EPS)
#EPS_PROB =PROB_TYPE(0.00000000001)
EPS_PROB =PROB_TYPE(1e-11)  #To be used inside the log function
EPS_LOG =PROB_TYPE(1e-11)
BIG_EPS = 1e-6  #for integer works and histogram
#needed: integer histogram

#decorators
def test_tdd(f):
    if TEST:
        print "Testing function "+f.__name__+"()...",
        f()
        print "passed"

#def test_log(x):
#    if VERBOSE:
#        print x

def enum(*sequential, **named):
    """ Usage:  Numbers = enum('ZERO', 'ONE', 'TWO') ; print Numbers.ZERO; print Numbers.reverse_mapping[2]"""
    #Based on: http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python    #By Alec Thomas
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    enums['UnknownExcecption'] = Exception
    return type('Enum', (), enums)

BiasType = enum("NAIVE0", "QE1", "NAIVE_CORRECTED2","PANZERI3","MONTEMURRO4","SADDLE5","SIMPS_QUAD6","BUB7","TYPE8","LATHAM")
print BiasType.NAIVE0, BiasType.QE1, BiasType.NAIVE_CORRECTED2,BiasType.PANZERI3,BiasType.MONTEMURRO4,BiasType.SADDLE5,\
BiasType.SIMPS_QUAD6,BiasType.BUB7,BiasType.TYPE8,BiasType.LATHAM



def range_shuffle(nta):
    """%this function shuffles the index matrix for all stimulus conditions
    :param nta:
    :type nta: list #of int
    :return: idx
    :rtype:
    """
    #function idx=range_shuffle(nt);
    m=max(nta) #??
    ns=len(nta)
    idx=np.zeros([ns,m],int) #np.zeros(ns,m); #todo: int type
    for s in range(ns): #s=1:ns
        #idx(s,1:nta(s))=randperm(nta(s));
        #np.random.shuffle(arr)
        #np.random.permutation(arr)
        #idx[s,0:(nta[s]-1)] = np.random.permutation(range(nta[s]))
        idx[s,0:nta[s]] = np.random.permutation(range(nta[s]))+1
    #end
    return idx

def range_frac(range1,nta,f,k):
    """This function extratcs subranges from the shuffled index matrix"""
    #todo: not tested
    #function r=range_frac(range1,nta,f,k);
    ns=len(nta)
    #new_nt=floor(nta/f);
    assert type(f) is int
    #todo: some trials are removed in QE
    new_nta = nta/f
    assert len(new_nta)==len(nta)
    m=max(new_nta) #m=max(new_nt);
    #r=zeros(ns,m);
    r=np.zeros([ns,m],int) #todo: int -> ?
    #for s=1:ns
    for s in range(ns):
        #r(s,1:new_nt(s))=range1(s,1+(k-1)*new_nt(s):k*new_nt(s));
        r[s,0:nta[s]] = range1[s, range((k-1)*new_nta[s], k*new_nta[s])  ] #todo: not tested
    #end
    return r

def lagrange3(x,y,xx):
    #todo: not tested
    #function px=lagrange3(x,y,xx)
    #   px=(xx-x(2))*(xx-x(3))/((x(1)-x(2))*(x(1)-x(3)))*y(1)+(xx-x(1))*(xx-x(3))/((x(2)-x(1))*(x(2)-x(3)))*y(2)+(xx-x(1))*(xx-x(2))/((x(3)-x(1))*(x(3)-x(2)))*y(3);
    return (xx-x[1])*(xx-x[2])/((x[0]-x[1])*(x[0]-x[2]))*y[0]+(xx-x[0])*(xx-x[2])/((x[1]-x[0])*(x[1]-x[2]))*y[1]+(xx-x[0])*(xx-x[1])/((x[2]-x[0])*(x[2]-x[1]))*y[2]

#def _debug_show_table(s1, rng):
#    rng=set(s1)
#    for i in rng:
#        c=len(filter(lambda x:x==int(i),s1))
#        if c>0:
#            print i,':',c
def _debug_show_table2(s1):
    #s2=map(lambda x: str(x),s1) #list(np.sort(s1.flatten())))
    #print ','.join(s2)
    rng=set(s1)
    for i in rng:
        c=len(filter(lambda x:x==int(i),s1))
        if c>0:
            print i,':',c
    print

@test_tdd
def test_range_shuffle():
    nta=[4,3,10,1,0]
    x=range_shuffle(nta)
    #print x
    #for i in range(x.shape[0]):
    #    print sum(x[i,:]>0)
    #print filter( lambda x: sum(x[i,:]>0), x )
    nta_reproduced = [sum(x[i,:]>0) for i in range(x.shape[0])]
    #test_log( np.sort(nta_reproduced) )
    #test_log( np.sort(nta) )
    assert all(np.sort(nta_reproduced) == np.sort(nta))

def _checktype_spk1LNS(spk, nta=None):
    assert type(spk) is np.ndarray
    #print type(spk[0,0,0,0]) #int64?!
    assert type(spk[0,0,0,0]) is RESP_TYPE #np.int8 #int
    ns=spk.shape[4-1] #ns=size(spk,4);
    ntr=spk.shape[3-1] #ntr=size(spk,3);
    L=spk.shape[2-1] #size(spk,2);
    if DEEP_TEST_DATA:
        M=max(spk.flatten())+1 #M=max(reshape(spk,1,[]))+1;
        assert min(spk.flatten())>=0
        #for i in range(1):
        #    for l in range(L):
        #        for nt in range(ntr):
        #            for s in range(ns):
        #                spk[i,l,nt,s] #spk[i,L,ntr,ns]
    if not nta is None:
        for i in range(1):
            for l in range(L):
                #for nt in range(nta):
                    for s in range(ns):
                        assert (spk[i,l,ntr:nta[s],s] == 0)
                        assert sum(abs(spk[i,l,ntr:nta[s],s])) == 0

def _checktype_nta(nta, ns):
    #assert type(nta) is np.ndarray or type(nta) is list
    #assert type(nta[0]) is COUNT_TYPE or type(nta[0]) is int #alwso allow plain Python list of int
    assert type(nta) is np.ndarray
    assert type(nta[0]) is COUNT_TYPE
    #alsochecks the consistency
    assert len(nta) == ns



#def _checktype_probr(p,...,nta,ns,...)

def probr(spk,nta,r,f, return_count=False):
    #function p=probr(spk,nt,r,f)
    """
    :param spk:
    :type spk: np.ndarray
    :param nta:
    :type nta: list
    :param r:
    :type r:
    :param f:
    :type f: int
    :return: p
    :rtype:
    this function works with the static version
    f is the factor that divides the number of trials to use
    nta stores the number of trials per stimulus condition
    """
    #rsa  LxNxS  1xLxNxS  1LNS spk1xLxNxS spk1LNS
    assert type(spk) is np.ndarray
    _checktype_spk1LNS(spk, nta=None)
    #assert len(spk.shape)==4
    ntr=spk.shape[3-1] #ntr=size(spk,3);
    ns=spk.shape[4-1] #ns=size(spk,4);
    L=spk.shape[2-1] #size(spk,2);
    M=max(spk.flatten())+1 #M=max(reshape(spk,1,[]))+1;
    _checktype_nta(nta,ns)

    #%number of trials for the subset
    assert type(f) is int
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

            trials[(i-1):(i-1+new_nta[s]-1+1),:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s] ) #.transpose()?

            #   #idx[s,0:(nta[s]-1)] = np.random.permutation(range(nta[s]))
            #   idx[s,0:nta[s]] = np.random.permutation(range(nta[s]))+1

            #i=i+new_nta(s);
            i += new_nta[s]
        #end
    else:
        #r[s,range(new_nta[s])] ==?
        for s in range(ns): #for s=1:ns
            #trials(i:i+new_nta(s)-1,:)=squeeze(spk(1,:,r(s,1:new_nta(s)),s));
            trials[i-1:i+new_nta(s)-1+1,:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s] ) #no transpose
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
    #print _debug_show_table2(wi.flatten()) #
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
    if return_count:
        count=count.astype(COUNT_TYPE)
        return p,count
    else:
        return p




def hr(spk,nta,biastype):
    #function [h0]=hr(spk,nt,biastype)
    """#his function estimates the response entropy of a set of trials
    The result is given in bits
    The estimator implemented is chosen by biastype
    :type biastype: BiasType
    """
    #global betac #???
    hc0=0
    hc1=0
    hc2=0
    hc3=0
    hc4=0
    hc5=0
    L=spk.shape[1] #L=size(spk,2)
    ntr=sum(nta) #%total number of trials
    ns=spk.shape[3] #size(spk,4); #ns=size(spk,4);
    #todo: Don't shuffle in the Naive estiamtion
    range1=range_shuffle(nta)
    #print range
    #print range
    p=probr(spk,nta,range1,1) #*********
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%Direct estimation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #print "-------"
    #print p
    hc0=-sum( p * np.log2(p + EPS_LOG)) #hc0=-sum((p) .* np.log2(p+ eps));
    if biastype==BiasType.NAIVE0:
        #switch biastype
        #case 0
        bias=0
        h0=hc0
        return h0

    elif biastype==BiasType.QE1:
        # case 1
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #   %range=range_shuffle(nt);
        r21=range_frac(range1,nta,2,1)
        r22=range_frac(range1,nta,2,2)
        r41=range_frac(range1,nta,4,1)
        r42=range_frac(range1,nta,4,2)
        r43=range_frac(range1,nta,4,3)
        r44=range_frac(range1,nta,4,4)

        p21=probr(spk,nta,r21,2)
        p22=probr(spk,nta,r22,2)
        p41=probr(spk,nta,r41,4)
        p42=probr(spk,nta,r42,4)
        p43=probr(spk,nta,r43,4)
        p44=probr(spk,nta,r44,4)
        
        h21=-sum(p21*np.log2(p21+EPS_LOG))
        h22=-sum(p22*np.log2(p22+EPS_LOG))
        h41=-sum(p41*np.log2(p41+EPS_LOG))
        h42=-sum(p42*np.log2(p42+EPS_LOG))
        h43=-sum(p43*np.log2(p43+EPS_LOG))
        h44=-sum(p44*np.log2(p44+EPS_LOG))
        h4=(h41+h42+h43+h44)/4
        h2=(h21+h22)/2
        
        n1=sum(nta)
        n2=sum(np.floor(nta/2))
        n4=sum(np.floor(nta/4))
        
        h0=lagrange3([1.0/n4, 1.0/n2, 1.0/n1],[h4,h2,hc0],0)
        #%h0=(8*hc0-6*h2+h4)/3; %parabolic extrapolation
        #%h0=(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hd*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
        
        #%hst=(4*hd-h21-h22)/2; %linear extrapolation
        return h0

"""

    elif biastype==2:
        #case 2
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Naive correction
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         
        bias0=(sum(p>EPS)-1)/(2*ntr*np.log(2)); #bias0=(sum(p>EPS)-1)/(2*ntr*log(2));
        n=p*ntr;
        pp=p;#%(n+1/ntr)./(ntr+1);
        t2=sum((((pp>EPS-pp).^2)./(pp+EPS))/(8 * ntr^2 *log(2)));
        ********
        h0=hc0+bias0+t2;
        print bias0
        print t2
        return h0
    elif biastype==3:
        #case 3
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Panzeri
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        R = bayescount(ntr,p);
        bias = (R-1)/(2*ntr*np.log(2));
        h0=hc0+bias;
        return h0

        
        
    elif biastype==4:
        #case 4
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Montemurro
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        idx=numpy.random.permutation(range(ntr)) #randperm(ntr);
        ntr2=np.round(ntr/2);

        **********
        r21=idx(1:ntr/2);
        r22=idx(ntr/2+1:end);
        p21=probr(spk,r21,M);
        p22=probr(spk,r22,M);
        h21=-sum(p21.*log2(p21+EPS_LOG));
        h22=-sum(p22.*log2(p22+EPS_LOG));
        bias0=(sum(p>EPS)-1)/(2*ntr*ns*log(2));
        bias1=(sum(p21>EPS)-1)/(ntr*ns*log(2));
        bias2=(sum(p22>EPS)-1)/(ntr*ns*log(2));
        h21=h21+bias1;
        h22=h22+bias2;
        h2=(h21+h22)/2;
        h00=hc0+bias0;
        h0=lagrange2([1/ntr2 1/ntr],[h2 h00],0);
        #%h0=(4*h00-h21-h22)/2;
               
    elif biastype==5:
        #case 5
        #%first recover absolute freqs
        n=round(p*ntr);
        [h0]=saddleentr3(n);

        
    elif biastype==6:
      #case 6
      #%first recover absolute freqs
      n=round(p*ntr);
      [h0]=simps_quad(n,betac(L,:));
    elif biastype==8:
        #case 8
        #%range=range_shuffle(nt);
        r21=range_frac(range,nta,2,1);
        r22=range_frac(range,nta,2,2);
        
        p21=probr(spk,nta,r21,2);
        p22=probr(spk,nta,r22,2);
        
        h21=-sum(p21.*log2(p21+EPS_LOG));
        h22=-sum(p22.*log2(p22+EPS_LOG));
        h2=(h21+h22)/2;
        n1=sum(nta);
        n2=sum(floor(nta/2));

        h0=lagrange2([1/n2 1/n1],[h2 hc0],0);
        
        
        
    #%  case 7
    #%  % the following is just an example
    #% N=ntr*ns;
    #% lin_vec=[1:1:N];
    #% a_vec(1)=0;
    #% a_vec(2:1:N+1) = [-(lin_vec/N).*log(lin_vec/N)];
    #% clear lin_vec;
    #% [bias_value,var_bound]=bub_bv_func(a_vec,p,0); % this provide things in nats
    #% clear a_vec;
    #% bias_value=bias_value/log(2);
    #% %var_bound=var_bound/(log(2)*log(2));
    #% h0=hc0-bias_value;
    #% 
    #% case 8
    #%     
    #% N=ntr*ns;
    #% R=length(p); % R is the size of the response space - if you have not defined it yet, it could be determined from the size of the probability vector as I did on the left 
    #% lin_vec=[1:1:N];
    #% a_me_vec(1)=0;
    #% a_me_vec(2:1:N+1) = [-(lin_vec/N).*log(lin_vec/N)];
    #% a_mm_vec(1) = -1/(2*N*R);
    #% a_mm_vec(2:1:N+1) = [(-(lin_vec/N).*log(lin_vec/N)) + (1-(1/R))/(2*N) ];
    #% clear lin_vec;
    #% [bias_value_me,var_bound_me]=bub_bv_func(a_me_vec,p,0); % this provide things in nats
    #% [bias_value_mm,var_bound_mm]=bub_bv_func(a_mm_vec,p,0); % this provide things in nats
    #% clear a_me_vec,a_mm_vec;
    #% bias_value_me=bias_value_me/log(2);
    #% bias_value_mm=bias_value_mm/log(2);
    #% %IMPORTANT NOTE:
    #% %the bias_value_me should be subtracted from the raw entropy estimate
    #% % the bias_value_mm shold be subtracted from the naive-corrected entropy estimate (the one obtained from the raw estimate subtracting the C_1 Panzeri bias term computed with the "naive" counting of bins)
    #% bias0=(sum(p>eps)-1)/(2*ntr*ns*log(2));  
    #% h0=hc0+bias0;
    #%    
    #%h0=h0-bias_value_mm;


        
        
#%error estimation, Latham's
#%N=ntr;
#%err=sqrt((sum(p.*log2(p+EPS_LOG).^2)-(hd*L)^2)/(L*N));
"""


def _test_data_spk_singleval(L,nta_arr, value):
    L=2; nta=np.array(nta_arr,COUNT_TYPE);ns=len(nta)
    _checktype_nta(nta, ns)
    spk = np.zeros([1,L,max(nta),ns], RESP_TYPE)
    for i in range(1):
        for l in range(L):
            for s in range(ns):
                #print ns, nta[s]
                for tr in range(nta[s]):
                    spk[i,l,tr,s] = RESP_TYPE(value)
    return spk,L,nta,ns

def _test_data_spk_rand(L,nta_arr, M):
    L=2; nta=np.array(nta_arr,COUNT_TYPE);ns=len(nta)
    _checktype_nta(nta, ns)
    #spk = np.zeros([1,L,len(nta),ns], np.int8)
    spk = np.zeros([1,L,max(nta),ns], RESP_TYPE)
    for i in range(1):
        for l in range(L):
            for s in range(ns):
                #print ns, nta[s]
                for tr in range(nta[s]):
                    spk[i,l,tr,s] = RESP_TYPE(np.random.random_integers(0,M-1))
                    #M=max(spk.flatten())+1 ==> between [0,M-1]
    return spk,L,nta,ns

@test_tdd
def test_hr_probr():
    #def make_spk1LNS():
    #    zeros()i,l,ntr:nta[s],s
    #    spk =
    spk,L,nta,ns = _test_data_spk_singleval(L=2,nta_arr=[1,2,3],value=1)

    range1=range_shuffle(nta)
    p,ctr=probr(spk,nta,range1,1, True)
    #print ctr,
    #print np.sort(ctr)
    A = np.sort(ctr) # [0,0,0,...,0, ntr]
    A[len(A)-1] == sum(nta)
    assert sum(abs(A[range(len(A)-1)]))==0

    h_R=hr(spk,nta,biastype=BiasType.NAIVE0)
    print 'H(R)=',abs(h_R), '=?=' , EPS_PROB #3.20342650381e-16 2.22045e-16
    assert abs(h_R) < EPS_PROB*100



    NTA = [1000,1000,1000] #[10,20,30]
    spk,L,nta,ns = _test_data_spk_rand(L=2,nta_arr=NTA,M=5)
    #print spk
    #s1=map(lambda x: str(x),list(np.sort(spk.flatten())))
    #print ','.join(s1)
    #_debug_show_table(s1,range(0,70))
    s1=list(np.sort(spk.flatten()))
    #_debug_show_table(s1,range(0,70))
    #_debug_show_table2(s1)
    #good

    range1=range_shuffle(nta)
    p,counts=probr(spk,nta,range1,1, True)
    #print counts
    #print counts
    #print -np.sort(-counts)

    #[226 145 136 135 130 128 125 124 124 121 120 120 119 119 118 117 117 115 115 113 113 112 108 100   0]
    #NOT GOOD!!

    #print np.sort(p)
    #A = np.sort(ctr) # [0,0,0,...,0, ntr]
    #A[len(A)-1] == sum(nta)
    #assert sum(abs(A[range(len(A)-1)]))==0

    h_R=hr(spk,nta,biastype=BiasType.NAIVE0)
    print '|H(R)|=',abs(h_R) , np.log2(5)*L,EPS_PROB #3.20342650381e-16 2.22045e-16
    #4.6389075846 bit!
    #assert abs(h_R) < EPS_PROB*100

@test_tdd
def test_hr_distr():
    M=5
    NTA = [100,100,100] # NTA= [10,20,30]  #[1000,1000,10]
    a=[]
    for tr in range(20):
        spk,L,nta,ns = _test_data_spk_rand(L=2,nta_arr=NTA,M=M)
        h=hr(spk,nta,biastype=BiasType.NAIVE0)
        #print h - np.log2(M)*L
        a.append(h)
    print np.mean(a) - np.log2(M)*L,',' , '+-', np.std(a)
    #What does the distribution look like?

@test_tdd
def test_hr_distr_QE():
    M=5
    NTA = [100,100,100] # NTA= [10,20,30]  #[1000,1000,10]
    a=[]
    for tr in range(20):
        spk,L,nta,ns = _test_data_spk_rand(L=2,nta_arr=NTA,M=M)
        h=hr(spk,nta,biastype=BiasType.QE1)
        #print 'H', h - np.log2(M)*L
        a.append(h)
    print np.mean(a) - np.log2(M)*L,',' , '+-', np.std(a)
    #What does the distribution look like?


#why??
#np.max([2,3,3.4])
#Out[14]: 3.3999999999999999
