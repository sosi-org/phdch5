
import numpy as np
#import scipy
#import matcompat

import matplotlib.pylab as plt

#
from hx_types import is_any_int_type



def pmarg4(p, id, iu, L, M):

    assert type(M) is int, repr(type(M))
    assert type(id) is int, repr(type(id))
    assert type(iu) is int, repr(type(iu))

    """
    u=iu-1;
    d=id-1;
    A=ones(M^(L-u-1),1)*[0:M^(d)-1];
    B=M^(u+1)*[0:M^(L-u-1)-1]'*ones(1,M^d);
    I0=1+reshape(A+B,[],1);
    pm=zeros(1,M^L);
    for j=0:M^(u-d+1)-1
        I=I0+M^d*j;
        pm(I)=sum(p(I));
    end
    """


    #%pass to 0:L-1 representation
    #%u=L-id;%bit's order is reverse of bin's order !!!
    #%d=L-iu;
    u = iu-1
    d = id-1
    #print type(u), type(d)
    #A = np.dot(np.ones(matixpower(M, L-u-1.), 1.), np.array(np.hstack((np.arange(0., (matixpower(M, d)-1.)+1)))))
    #A = np.ones([M**(L-u-1), 1],int).dot( np.arange(0, (M**d)-1+1,int).reshape((1,M**d)))
    zz = np.array(range(0, (M**d)-1+1), int).reshape((1,M**d))
    A = np.ones([M**(L-u-1), 1],int).dot( zz)
    #B = np.dot(np.dot(matixpower(M, u+1.), np.array(np.hstack((np.arange(0., (matixpower(M, L-u-1.)-1.)+1)))).conj().T), np.ones(1., matixpower(M, d)))
    #B = (M**(u+1)) *  np.arange(0,M**(L-u-1)-1+1,int).reshape([M**(L-u-1),1]).dot(np.ones([1, M**d].int))
    yy=np.array(range(0,M**(L-u-1)-1+1),int).reshape([M**(L-u-1),1])
    B = (M**(u+1)) *  yy.dot(np.ones([1, M**d],int))
    #I0 = 1.+np.reshape((A+B), np.array([]), 1.)
    #I0 = 1.+np.reshape((A+B), [A.size,1])
    I0 = 1+(A+B).flatten()
    #pm = np.zeros((1, M**L))
    pm = np.zeros(M**L)
    for j in range(0, M**(u-d+1)-1+1):
        assert is_any_int_type(I0[0])
        I = I0+(M**d) * j
        assert is_any_int_type(I[0])
        #print I[0]
        #pm[int(I)-1] = np.sum(p[int(I)-1])
        assert max(I-1)<p.size
        assert max(I-1)<pm.size
        assert min(I-1)>=0
        #print max(I-1)
        #print min(I-1)
        pm[I-1] = np.sum(p[I-1])
        #np.sum(p[I-1]) should be scalar

    return pm

    # Local Variables: A, B, d, I, iu, j, M, L, p, u, I0, id, pm
    # Function calls: reshape, sum, ones, zeros, pmarg4
