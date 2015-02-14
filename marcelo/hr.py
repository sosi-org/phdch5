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
        print "Testing function "+f.__name__+"()..."
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
    maxntr=spk.shape[3-1] #ntr=size(spk,3);
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
                        #assert (spk[i,l,ntr:nta[s],s] == 0)
                        #assert sum(abs(spk[i,l,ntr:nta[s],s])) == 0
                        #assert (spk[i,l,nta[s]:maxntr,s] == 0)
                        assert sum(abs(spk[i,l,nta[s]:maxntr,s])) == 0

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

            #trials[(i-1):(i-1+new_nta[s]-1+1),:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s] ) #.transpose()?
            #trials[(i-1):(i-1+new_nta[s]-1+1),:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s], [0,3]) #.transpose()?
            trials[(i-1):(i+new_nta[s]-1),:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s], [0,3]) #.transpose()?

            #   #idx[s,0:(nta[s]-1)] = np.random.permutation(range(nta[s]))
            #   idx[s,0:nta[s]] = np.random.permutation(range(nta[s]))+1

            #i=i+new_nta(s);
            i += new_nta[s]
        #end
    else:
        #r[s,range(new_nta[s])] ==?
        for s in range(ns): #for s=1:ns
            #trials(i:i+new_nta(s)-1,:)=squeeze(spk(1,:,r(s,1:new_nta(s)),s));
            #trials[i-1:i+new_nta(s)-1+1,:] = np.squeeze( spk[0,:,r[s,range(new_nta[s])]-1,s] , [0,3] ) #no transpose
            #print new_nta[s]
            #print trials[(i-1):(i+new_nta[s]-1),0].shape
            A=spk[0,:,r[s,range(new_nta[s])]-1,s]
            #print A.shape #(1000,1)
            A2=np.squeeze(A) #A2=np.squeeze( A , [0,3] )
            trials[(i-1):(i+new_nta[s]-1),0] = A2 #no transpose
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
    #hc0=0
    #hc1=0
    #hc2=0
    #hc3=0
    #hc4=0
    #hc5=0
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
    nta=np.array(nta_arr,COUNT_TYPE);ns=len(nta)
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


def _test_data_spk_plain(pa,L,nta_arr):
    M=len(pa)
    nta=np.array(nta_arr,COUNT_TYPE);ns=len(nta)
    _checktype_nta(nta, ns)
    spk = np.zeros([1,L,max(nta),ns], RESP_TYPE)
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
                    spk[i,l,tr,s] = RESP_TYPE(R)


    return spk,L,nta,ns

def _test_data_spk_rand_pa(pa,L,nta_arr, M):
    #todo: based on _test_data_spk_rand
    nta=np.array(nta_arr,COUNT_TYPE);ns=len(nta)
    _checktype_nta(nta, ns)
    spk = np.zeros([1,L,max(nta),ns], RESP_TYPE)
    for i in range(1):
        for l in range(L):
            for s in range(ns):
                for tr in range(nta[s]):
                    spk[i,l,tr,s] = 0 #RESP_TYPE(np.random.random_integers(0,M-1))***
                    raise Exception("Not implemented")
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


###############
#   H(R|S)    #
###############

def probrs(spk,r,s,M, return_count=False):
    #function p=probrs(spk,r,s,M)
    """this function works with the static version
    f is the factor that divides the number of trials to use"""
    ntr=len(r)

    ns=spk.shape[4-1]
    L=spk.shape[2-1]

    #ntr=length(r);
    #ns=size(spk,4);
    #L=size(spk,2);
    #spkt=squeeze(spk(1,:,r,s));
    #print spk.shape
    if False:
        print r #[1,2,3,...,100]
        print s,len(r) #1
        print spk.shape
        print spk[0,:,r-1,s-1].shape
        print (spk[0,:,r-1,s-1]).shape #100,2 ??
    #spkt=spk[0,:,r-1,s-1].squeeze([0,3])
    spkt=spk[0,:,r-1,s-1] #no need???? #why ?? (100 x 2)

    #    #if L==1:
    #    #    spkt=spkt';
    #    trials=spkt.transpose() #trials=spkt';   #%trials=(reshape(spkt,L,[]))';
    trials=spkt #no need to transpose. why??


    #p=np.zeros([1,M^L],...)  #p=zeros(1,M^L);
    #count=zeros([M^L,1],...)
    #wi=1+trials*(M.^[0:L-1])';
    B = np.power(M,np.array(range(L)).reshape([L,1]))  #'; tranpose
    if False:
        print trials.shape
        print B.shape
    wi=1+trials.dot(B)

    #count=histc(wi,[1:M^L+eps]);
    maxrange = int(BIG_EPS+math.pow(M,L))
    edges = np.array(range(0,maxrange+1)) + 0.1
    count,e2 = np.histogram(wi.flatten(), edges) #
    #p=(count'/sum(count));
    p=count / PROB_TYPE(sum(count))
    if return_count:
        count=count.astype(COUNT_TYPE)
        return p,count
    else:
        return p

def hrs(spk,nta,biastype):
    #function [h0]=hrs(spk,nt,biastype)
    """It will estimate the entropy of a binary chain giving the result in bits per bin"""
    #%trials=squeeze(spk(1,:,:))';

    _checktype_spk1LNS(spk,nta)

    #%Bias correction

    #global betac  #todo: ???
    #hc0=0
    #hc1=0
    #hc2=0
    #hc3=0
    #hc4=0
    #hc5=0

    #%ntr=size(spk,3);
    #ns=size(spk,4);
    #L=size(spk,2);
    h0=0
    err=0

    L=spk.shape[1]
    ns=spk.shape[3]



    #M=1+max(reshape(spk,1,[]));
    M=1+max(spk.flatten())
    if M==1:
        M=2
    #end



    #for t=1:ns  #%over all stimulus conditions
    for t0 in range(0,ns):
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%Direct estimation
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        ntr=nta[t0] #ntr=nt(t);
        #%trials=(squeeze(spk(1,:,:,t)))'; %trials set for current stimulus condition
        #prs=probrs(spk,[1:ntr],t,M);  %it uses the same probr functions as Hr does!
        prs = probrs(spk,np.array(range(1,ntr+1)).astype(int),t0+1,M) #%it uses the same probr functions as Hr does!

        #hdt=-sum(prs.*log2(prs+eps))*nt(t);
        hdt = -sum(prs * np.log2(prs+EPS_LOG))*nta[t0]



        if biastype==0:
            #case 0
            bias=0
            h0 += hdt
        elif biastype==1:
        #case 1
            #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
            #%idx=randperm(ntr); %it performs a random permutation of the indeces to trials
            #%divide in ranges
            #idx=randperm(ntr);
            idx=np.random.permutation(range(ntr))+1 #idx=randperm(ntr);******
            ntr2=int(math.floor(ntr/2))
            ntr4=int(math.floor(ntr/4))
            r21=idx[range(ntr2)] #r21=idx(1:ntr2);

            # (i,j) ---> range(i-1,j)
            #r22=idx(ntr2+1:2*ntr2);
            r22=idx[range(ntr2+1-1,2*ntr2)]

            r41=idx[range(ntr4)] #r41=idx(1:ntr4);
            r42=idx[range(ntr4,2*ntr4)]  #r42=idx(ntr4+1:2*ntr4);
            r43=idx[range(2*ntr4,3*ntr4)] #r43=idx(2*ntr4+1:3*ntr4);
            r44=idx[range(3*ntr4,4*ntr4)] #r44=idx(3*ntr4+1:4*ntr4);

            p21=probrs(spk,r21,t0+1,M)
            p22=probrs(spk,r22,t0+1,M)
            p41=probrs(spk,r41,t0+1,M)
            p42=probrs(spk,r42,t0+1,M)
            p43=probrs(spk,r43,t0+1,M)
            p44=probrs(spk,r44,t0+1,M)
            h21=-sum(p21*np.log2(p21+EPS_LOG))
            h22=-sum(p22*np.log2(p22+EPS_LOG))
            h41=-sum(p41*np.log2(p41+EPS_LOG))
            h42=-sum(p42*np.log2(p42+EPS_LOG))
            h43=-sum(p43*np.log2(p43+EPS_LOG))
            h44=-sum(p44*np.log2(p44+EPS_LOG))
            h4=nta[t0]*(h41+h42+h43+h44)/4.0
            h2=nta[t0]*(h21+h22)/2.0

            #h0=h0+lagrange3([1/ntr4 1/ntr2 1/ntr],[h4 h2 hdt],0);
            h0 += lagrange3([1.0/ntr4, 1.0/ntr2, 1.0/ntr],[h4, h2, hdt],0)
            #%h0=h0+(8*hdt-6*h2+h4)/3;
            #%hs1=hs1+(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hdt*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
            #%hst=hst+(4*hdt-h21-h22)/2;

        elif  biastype==2:
            #case 2
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            #%Naive
            #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            bias0=(sum(prs>EPS_PROB)-1)/(2*nta[t0]*np.log(2.0))
            h0 += hdt+bias0*nta[t0] #h0=h0+hdt+bias0*nt(t);

        """
            case 3
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Panzeri
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            R = bayescount(nt(t),prs);
            bias = (R-1);
            h0=h0+hdt+nt(t)*bias/((2*nt(t)*log(2)));

            case 4
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %Montemurro
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            idx=randperm(ntr); %it performs a random permutation of the indeces to trials
            %divide in ranges
            ntr2=round(ntr/2);
            ntr4=round(ntr/4);
            r21=idx(1:ntr2);
            r22=idx(ntr2+1:end);
            r41=idx(1:ntr4);
            r42=idx(ntr4+1:ntr2);
            r43=idx(ntr2+1:3*ntr4);
            r44=idx(3*ntr4+1:end);

            p21=probrs(spk,r21,t);
            p22=probrs(spk,r22,t);
            h21=-sum(p21.*log2(p21+eps));
            h22=-sum(p22.*log2(p22+eps));

            bias0=(sum(prs>eps)-1)/(2*ntr*log(2));
            bias1=(sum(p21>eps)-1)/(ntr*log(2));
            bias2=(sum(p22>eps)-1)/(ntr*log(2));
            h21=h21+bias1;
            h22=h22+bias2;
            h2=(h21+h22)/2;
            h00=hdt+bias0;
            %   h0=h0+(4*h00-h21-h22)/2;
            h0=h0+lagrange2([1/ntr2 1/ntr],[h2 h00],0);

            case 5
            %Nemmenman
            %first recover absolute freqs
            n=round(prs*nt(t));
            [hnemt]=saddleentr3(n);
            h0=h0+hnemt*nt(t);

            case 6
            %Nemmenman
            %first recover absolute freqs
            n=round(prs*nt(t));
            K=M^L;
            [hnemt]=simps_quad(n,betac(K,:));
            h0=h0+hnemt*nt(t);

            case 8
            ntr=nt(t);
            idx=randperm(ntr);
            ntr2=floor(ntr/2);
            r21=idx(1:ntr2);
            r22=idx(ntr2+1:2*ntr2);

            p21=probrs(spk,r21,t,M);
            p22=probrs(spk,r22,t,M);
            h21=-sum(p21.*log2(p21+eps));
            h22=-sum(p22.*log2(p22+eps));
            h2=nt(t)*(h21+h22)/2;
            h0=h0+lagrange2([1/ntr2 1/ntr],[h2 hdt],0);
            %h0=h0+(8*hdt-6*h2+h4)/3;
            %hs1=hs1+(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hdt*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));


            % case 7
            % lin_vec=[1:1:ntr];
            % a_vec(1)=0;
            % a_vec(2:1:ntr+1) = [-(lin_vec/ntr).*log(lin_vec/ntr)];
            % clear lin_vec;
            % [bias_value,var_bound]=bub_bv_func(a_vec,prs,0); % this provide things in nats
            % clear a_vec;
            % bias_value=bias_value/log(2);
            % %var_bound=var_bound/(log(2)*log(2));
            % h0=h0+hdt-bias_value;
            %
            % case 8
            %
            % N=ntr;
            % R=length(prs); % R is the size of the response space - if you have not defined it yet, it could be determined from the size of the probability vector as I did on the left
            % lin_vec=[1:1:N];
            % a_me_vec(1)=0;
            % a_me_vec(2:1:N+1) = [-(lin_vec/N).*log(lin_vec/N)];
            % a_mm_vec(1) = -1/(2*N*R);
            % a_mm_vec(2:1:N+1) = [(-(lin_vec/N).*log(lin_vec/N)) + (1-(1/R))/(2*N) ];
            % clear lin_vec;
            % [bias_value_me,var_bound_me]=bub_bv_func(a_me_vec,prs,0); % this provide things in nats
            % [bias_value_mm,var_bound_mm]=bub_bv_func(a_mm_vec,prs,0); % this provide things in nats
            % clear a_me_vec,a_mm_vec;
            %
            % bias_value_me=bias_value_me/log(2);
            % bias_value_mm=bias_value_mm/log(2);
            % %IMPORTANT NOTE:
            % %the bias_value_me should be subtracted from the raw entropy estimate
            % % the bias_value_mm shold be subtracted from the naive-corrected entropy estimate (the one obtained from the raw estimate subtracting the C_1 Panzeri bias term computed with the "naive" counting of bins)
            %
            % bias0=(sum(prs>eps)-1)/(2*ntr*log(2));
            % h0=h0+hdt+bias0;
            % h0=h0-bias_value_mm;
        """
        #end  %swithc

    #end #for

    h0 = h0/sum(nta)
    return h0



@test_tdd
def test_hrs_distr_twobiases():
    for btype in [0,1]:
        M=5
        NTA = [1000,1000,1000]
        a=[]
        for tr in range(20):
            spk,L,nta,ns = _test_data_spk_rand(L=2,nta_arr=NTA,M=M)
            h=hrs(spk,nta,biastype=btype)
            #print 'H', h - np.log2(M)*L
            a.append(h)
        print np.mean(a) - np.log2(M)*L,',' , '+-', np.std(a)
        #What does the distribution look like?


@test_tdd
def test_MutualInformation_0_distr_twobiases():
    for btype in [0,1]:
        M=5
        NTA = [1000,1000,1000]
        a=[]
        for tr in range(20):
            spk,L,nta,ns = _test_data_spk_rand(L=2,nta_arr=NTA,M=M)
            h2=hrs(spk,nta,biastype=btype)
            h1=hr(spk,nta,biastype=btype)
            #print 'H', h1-h2 #h - np.log2(M)*L
            a.append(h1-h2)
        #print np.mean(a) - np.log2(M)*L,',' , '+-', np.std(a)
        print np.mean(a) - 0.0,',' , '+-', np.std(a)
        #What does the distribution look like?

def H_from_p(p):
    print sum(p)
    assert abs(sum(p) - 1.0) < EPS_PROB
    assert len(p.shape)==1
    return - sum(p * np.log2(p + EPS_LOG))

@test_tdd
def test_MutualInformation_distr_twobiases():
    L=1
    #PA=[0.1,0.2,0.7]
    #PA=[1.0/3,1.0/3,1.0/3]
    NS=2
    PA=np.array([1.0,1.0]); PA=PA/np.sum(PA)
    p1 = (np.tile(np.array(PA).reshape([NS,1]),[1,NS]) /2 + np.eye(NS) / 2.0) / NS
    px=np.sum(p1,axis=0)
    py=np.sum(p1,axis=1)
    pxy=p1.flatten()
    #assert sum(px)==1.0
    #assert sum(py)==1.0
    #assert sum(pxy)==1.0
    h1=H_from_p(px)
    h2=H_from_p(py)
    h3=H_from_p(pxy)
    #print H_from_p(px) + H_from_p(py) - H_from_p(pxy)
    #print h1
    #print h2
    #print h3
    print "analytical: ",(h1+h2-h3) * L

    for btype in [0,1]:
        #M=5
        #NTA = [10000,10000,10000]
        NTA = [1000,1000]
        assert np.sum(PA)==1.0
        a=[]
        for tr in range(20):
            #spk,L,nta,ns = _test_data_spk_rand_pa(pa, L=2,nta_arr=NTA,M=M)
            sys.stdout.write("-");sys.stdout.flush()
            spk,L,nta,ns = _test_data_spk_plain(PA, L=L,nta_arr=NTA)
            sys.stdout.write("_");sys.stdout.flush()
            h2=hrs(spk,nta,biastype=btype)
            h1=hr(spk,nta,biastype=btype)
            #sys.stdout.write("_");sys.stdout.flush()
            #print 'H', h1-h2 #h - np.log2(M)*L
            #_debug_show_table2(spk.flatten())

            a.append(h1-h2)
        #print np.mean(a) - np.log2(M)*L,',' , '+-', np.std(a)
        print np.mean(a) - 0.0,',' , '+-', np.std(a)
        #What does the distribution look like?

#Testing function test_MutualInformation_distr_twobiases()...
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_0.673389327746 , +- 0.0162120973153
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_0.659461452067 , +- 0.0177021684313
#passed
#
#0.775836103726


#1.0
#analytical:  0.666666666753
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_0.579919387054 , +- 0.00653052998022
#-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_0.581752993395 , +- 0.00475973942452
#passed
#sohail@ss-desktop:~/sohail/sig/marcelo$

"""
#10000
analytical:  0.377443751082
-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_0.330853238447 , +- 0.00535535147218
-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_0.332011461827 , +- 0.004686515463
passed
"""