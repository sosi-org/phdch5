
import numpy as np
#import scipy
#import matcompat

#import matplotlib.pylab as plt
#from regular_discretize import regular_discretize
from edges_centres import edges_centres
from whichbin import whichbin



from hx_types import is_any_int_type
def dith_unsure(y, centres):
    #def dith(y, centres):  -> is it renamed?
    #print y.shape
    #print "============"

    # Local Variables: overall_mean, HOW_MANY_STANDARD_DEVIATIONS, d, dims, i, v, M, sample, DEBUG, ii, edges, m1, edges1, overall_sigma, centres, y, z, remainder, DITHINFO, implied
    # Function calls: disp, max, unique, isscalar, Inf, arreq, min, mean, nargout, sqrt, whichbin, length, zeros, sprintf, error, var, diff, edges_centres, prod, dith, size
    #% usage: z=dith(y,M);
    #% usage: z=dith(y,centers); %centres must be an array of scalars
    #% size(y)= (trials x length)
    #% size(y) = trials
    #% z will not contain any 0 (indices start with 1)
    DEBUG = 1.
    #%turn off the slow run-time tests
    #%usage: dith(y,M)   %M=arg centres
    #if len(centres) == 1:
    #    M = centres
    #    if not type(M) is int:
    #        #if np.prod(matcompat.size(M)) != 1.:
    #        rise Exception('usage: z=dith(y,M);')
    if type(centres) is int:
        M = centres
        
        #dims = np.arange(1., (length(matcompat.size(y)))+1)
        #dims = np.arange(1., (len(matcompat.size(y)))+1)
        #dims = np.arange(1, (len(matcompat.size(y)))+1)
        #not used anymore

        #%the default order is 2,1,3,4,5,6,7 ...
        #if length(dims) > 1.:
        #    dims[0:2.] = np.array(np.hstack((2., 1.)))
        
        
        #v = np.var(y, np.array([]), dims[0])
        #if length(dims) > 1.:
        #    for d in dims[1:]:
        #        v = np.mean(v, d)
        v=np.var(y.flatten())        
        
        
        overall_sigma = np.sqrt(v)
        #%assert(overall_sigma ~= 0.0, 'var is 0.0');
        #m1 = np.mean(y, dims[0])
        #if length(dims) > 1.:
        #    for d in dims[1:]:
        #        m1 = np.mean(m1, d)
        m1 = np.mean(y.flatten())
                
        
        overall_mean = m1
        HOW_MANY_STANDARD_DEVIATIONS = 2.
        #%just a scale of edges & centres value
        np.disp('using %d standard deviations, M=%d levels'%(HOW_MANY_STANDARD_DEVIATIONS, M))
        #%edges =   "values used for discretisation"
        #%centres2 =  "implied" value by each discretized level (used in dithering)"
        (edges, centres) = edges_centres(M, HOW_MANY_STANDARD_DEVIATIONS, overall_sigma, overall_mean)
        assert(M>1)
    else:
        #%usage: dith(y,centres)
        #%ds=centres(2)-centres(1);
        #edges1=centres(1:end-1) + diff(centres)/2;
        edges1 = centres[0:-1]+np.diff(centres) / 2.
        #edges1 = centres[0:0-1]+np.diff(centres)/2.
        #    edges=[-Inf,edges1,Inf];
        edges = np.array(np.hstack((-np.Inf, edges1, np.Inf)))
        M = len(centres)

        
    
    #%return extra information about discritization
    if True: #if nargout == 2.:
        DITHINFO = {} #np.array([])
        DITHINFO['edges'] = edges
        DITHINFO['centres'] = centres
        #%DITHINFO.z2= regular_discretize(y,M, ds, edges(2));
    
    
    #% if simple/regular agorithm
    #% z= regular_discretize(y,M, ds, edges(2));
    #%

    #%the actual slow calculations

    #print type(y)
    sz=y.shape    
    #%both edges & centres are needed.
    #z = np.zeros(y.shape, int)

    z = np.zeros((y.size,), int)
    y1=y.flatten(1)

    remainder = 0.
    #%initial remainder = 0 (i.e., last error, or remainder error)
    for i0 in range(y1.size): #np.arange(1., (length(z.flatten(1)))+1):
        sample = y1[i0]
        i=i0+1
        #%ii=find(histc(sample + remainder,edges));      %, 'first'
        ii = whichbin((sample+remainder), edges)
        implied = centres[ii-1]
        remainder = remainder+sample-implied
        #%remainder = remainder * 0.999;
        #print 'ii,i',(ii,i),  '   z:',z.shape,'  y1.size',y1.shape
        z[i-1] = ii
        if DEBUG:
            #%if isempty(ii) warning('error in discretization'); end
            #%if ii==0  error('find(histc()) not found');   end
            #%assert(~isempty(ii), 'error in discretization' );
            #%%assert(ii>0, 'find(histc(value)) did not fit into any interval');
            #%assert(length(ii)==1, 'histc porblem');
            pass

        
        
    #%warning: uses the last remainder for the next trial
    if DEBUG:
        #%slow ones here
        #%unique(z(:))'
        #%slow
        pass

    z=z.reshape(sz)       
    #print z.shape
    #print sz
    #print "***********************"
    #print type(z[0,0])
    assert is_any_int_type(z[0,0])
    #%inner function
    return z, DITHINFO

"""
#redundant!
def regular_discretize(y, M, ds, y0):

    # Local Variables: y, y0, M, ds, z2
    # Function calls: size, zeros, floor, regular_discretize
    z2 = np.zeros(matcompat.size(y),int)
    z2[:] = np.floor(matdiv(y.flatten(1)-y0, ds))+1.0.
    #%edges(J) ... +J-1
    #%out of bounds: oob = sum(spk2(:)>M) + sum((spk2(:)<0));
    z2[:] = z2.flatten(1)*(z2.flatten(1) > 0.)
    z2[:] = z2.flatten(1)+1.
    #%range: 1:M
    z2[:] = np.dot(z2.flatten(1) > M, M)+(z2.flatten(1)<=M)*z2.flatten(1)
    return z2

def edges_centres(M, HOW_MANY_STANDARD_DEVIATIONS, sigma, mn):

    # Local Variables: ds_i, HOW_MANY_STANDARD_DEVIATIONS, centres2, mn, M, edges, edges0, del, edges0_i, sigma
    # Function calls: length, edges_centres, Inf
    #%        ds=2*HOW_MANY_STANDARD_DEVIATIONS*sigma/M;
    #%        edges0=[-HOW_MANY_STANDARD_DEVIATIONS *sigma ...
    #%            : ds : ...
    #%            HOW_MANY_STANDARD_DEVIATIONS *sigma] ...
    #%            + mn;
    ds_i = matdiv(2.*HOW_MANY_STANDARD_DEVIATIONS, M)
    edges0_i = np.arange(-HOW_MANY_STANDARD_DEVIATIONS, (HOW_MANY_STANDARD_DEVIATIONS)+(ds_i), ds_i)
    edges0 = np.dot(edges0_i, sigma)+mn
    edges = edges0
    edges[int(np.array(np.hstack((1., 0))))-1] = np.array(np.hstack((-Inf, Inf)))
    #%first and last are replaced with Inf
    #%setting the levels (implied) values
    #%  edges : thresholds for cutting
    #%  centres2 : implied values
    #%
    if M == 2.:
        del = 0.
        #%centres2=[-ds/2,ds/2]; %was [0]
        centres2 = np.dot(np.array(np.hstack((-ds_i/2., ds_i/2.))), sigma)+mn
    else:
        #%edges = middle of each interval
        #%delt=diff(edges);
        #%delt=delt(2:(end-1));
        #%delt=delt(1);
        #%delt = edges(3)-edges(2); %becasue edges(1) is -Inf
        #%centres2=edges;
        #%centres2(1)=edges(2)-delt;
        #%centres2(end)=edges(end-1)+delt;
        centres2 = edges0
        #%before replacing -Inf , +Inf
        #centres2 = (centres2(2:end) + centres2(1:end-1))/2;
        centres2 = (centres2[1:]+centres2[0:0-1])/2.
        #%must contain 'M' levels
        
    
    #%previous procedure
    #%        overall_var =mean(var(y,[],2),1); % over time and then over trials.
    #%
    #%        overall_sigma=sqrt(overall_var);
    #%        overall_mean = mean(mean(y,2),1);
    #%        ds=2*2*overall_sigma/M; %
    #%        %for one channel only:
    #%        edges=[-HOW_MANY_STANDARD_DEVIATIONS *overall_sigma : ds : HOW_MANY_STANDARD_DEVIATIONS *overall_sigma] + overall_mean; %!!!
    #%        edges([1,end]) = [-Inf,Inf];
    #%        %edges = edges of the intervals
    #%no need for this function!
    #%function r=isscalar(a)
    #%    r= arreq(size(a) , [1,1]);
    #%end
    #%function r=andall(a)
    #%r=prod(a+0.0);
    #%end
    return edges, centres2
"""
def arreq(a, b):

    # Local Variables: a, r, b
    # Function calls: sum, abs, arreq
    r = np.sum(np.abs((a-b))) == 0
    return r