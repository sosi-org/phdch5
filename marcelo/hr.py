#!/usr/bin/python
# A port of Marcelo A Montemurro's code into Python
import numpy as np
import sys

VERBOSE = 0
#decorators
def test_tdd(f):
    print "Testing function "+f.__name__+"()...",
    f()
    print "passed"

def test_log(x):
    if VERBOSE:
        print x

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
    idx=np.zeros([ns,m],int) #np.zeros(ns,m);
    for s in range(ns): #s=1:ns
        #idx(s,1:nta(s))=randperm(nta(s));
        #np.random.shuffle(arr)
        #np.random.permutation(arr)
        #idx[s,0:(nta[s]-1)] = np.random.permutation(range(nta[s]))
        idx[s,0:nta[s]] = np.random.permutation(range(nta[s]))+1
    return idx

@test_tdd
def test_range_shuffle():
    nta=[4,3,10,1,0]
    x=range_shuffle(nta)
    #print x
    #for i in range(x.shape[0]):
    #    print sum(x[i,:]>0)
    #print filter( lambda x: sum(x[i,:]>0), x )
    nta_reproduced = [sum(x[i,:]>0) for i in range(x.shape[0])]
    test_log( np.sort(nta_reproduced) )
    test_log( np.sort(nta) )
    assert all(np.sort(nta_reproduced) == np.sort(nta))

exit(0)

"""
def hr(spk,nt,biastype):
    #function [h0]=hr(spk,nt,biastype)
    #This function estimates the response entropy of a set of trials
    #The result is given in bits
    #The estimator implemented is chosen by biastype

    #global betac #???
    hc0=0
    hc1=0
    hc2=0
    hc3=0
    hc4=0
    hc5=0
    L=spk.shape[1] #L=size(spk,2)
    ntr=sum(nt) #%total number of trials
    ns=spk.shape[3] #size(spk,4); #ns=size(spk,4);
    range=range_shuffle(nt);
    p=probr(spk,nt,range,1); #*********
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #%Direct estimation
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #np.emath.
    EPS = sys.float_info.epsilon  #http://stackoverflow.com/questions/23190017/is-pythons-epsilon-value-correct
    print EPS
    hc0=-sum( p * np.log2(p+ EPS)) #hc0=-sum((p) .* np.log2(p+ eps));
    if biastype==0:
        #switch biastype
        #case 0
        bias=0;
        h0=hc0;
        return h0
    elif biastype==1:
        # case 1
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #%This is the 3 point extrapolation taking 1/4, 1/2 and 1/1 of the trials
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #   %range=range_shuffle(nt);
        r21=range_frac(range,nt,2,1);
        r22=range_frac(range,nt,2,2);
        r41=range_frac(range,nt,4,1);
        r42=range_frac(range,nt,4,2);
        r43=range_frac(range,nt,4,3);
        r44=range_frac(range,nt,4,4);
        
        p21=probr(spk,nt,r21,2);
        p22=probr(spk,nt,r22,2);
        p41=probr(spk,nt,r41,4);
        p42=probr(spk,nt,r42,4);
        p43=probr(spk,nt,r43,4);
        p44=probr(spk,nt,r44,4);
        
        h21=-sum(p21*np.log2(p21+EPS));
        h22=-sum(p22*np.log2(p22+EPS));
        h41=-sum(p41*np.log2(p41+EPS));
        h42=-sum(p42*np.log2(p42+EPS));
        h43=-sum(p43*np.log2(p43+EPS));
        h44=-sum(p44*np.log2(p44+EPS));
        h4=(h41+h42+h43+h44)/4;
        h2=(h21+h22)/2;
        
        n1=sum(nt);
        n2=sum(np.floor(nt/2));
        n4=sum(np.floor(nt/4));
        
        h0=lagrange3([1.0/n4 1.0/n2 1.0/n1],[h4 h2 hc0],0);
        #%h0=(8*hc0-6*h2+h4)/3; %parabolic extrapolation
        #%h0=(-h2*ntr2^2*(ntr-ntr4)+h4*ntr4^2*(ntr-ntr4)+hd*ntr^2*(ntr2-ntr4))/((ntr-ntr2)*(ntr-ntr4)*(ntr2-ntr4));
        
        #%hst=(4*hd-h21-h22)/2; %linear extrapolation
        return h0

        
        
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
        h21=-sum(p21.*log2(p21+EPS));
        h22=-sum(p22.*log2(p22+EPS));
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
        r21=range_frac(range,nt,2,1);
        r22=range_frac(range,nt,2,2);
        
        p21=probr(spk,nt,r21,2);
        p22=probr(spk,nt,r22,2);
        
        h21=-sum(p21.*log2(p21+EPS));
        h22=-sum(p22.*log2(p22+EPS));
        h2=(h21+h22)/2;
        n1=sum(nt);
        n2=sum(floor(nt/2));

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
#%err=sqrt((sum(p.*log2(p+eps).^2)-(hd*L)^2)/(L*N));


def test_hr():
    pass


#why??
#np.max([2,3,3.4])
#Out[14]: 3.3999999999999999
"""