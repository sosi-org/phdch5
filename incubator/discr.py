
import numpy as np
#import scipy
#import matcompat

# if available import pylab (from matlibplot)
import matplotlib.pylab as plt

from edges_centres import edges_centres
from regular_discretize import regular_discretize

def discr(y, M):

    # Local Variables: overall_mean, dinfo, HOW_MANY_STANDARD_DEVIATIONS, centres, M, dims, edges, m1, overall_sigma, v, y, z, ds, d
    # Function calls: disp, isscalar, mean, regular_discretize, discr, sqrt, nargout, length, sprintf, error, var, edges_centres, prod, size
    #%usage: dith(y,M)
    #if np.prod(list(M.shape)) != 1.:
    #    raise Exception('usage: z=dith(y,M);')
    assert type(M) is int
    
    
    #print repr(y)
    #print repr(y.shape)
 
    #    #dims = np.array(np.hstack((2., 1.)))
    #    #dims=[2-1,1-1]
    #    dims=[0]
    #    #v = np.var(y, np.array([]), dims[0])
    #    v = np.var(y, axis=dims[0])
    #    if len(dims) > 1:
    #        for d in dims[1:]:
    #            v = np.mean(v, axis=d)
    #    #assert v is scalar
    #    print repr(v.shape)
    #    v=v.flatten()
    v=np.var(y.flatten())

    #print 'y.shape', y
    #print 'v.shape', v
            
    
    
    overall_sigma = np.sqrt(v)
    #print "v.shape", v.shape
    # #%assert(overall_sigma ~= 0.0, 'var is 0.0');
    #m1 = np.mean(y, axis=dims[0])
    #if len(dims) > 1:
    #    for d in dims[1:]:
    #        m1 = np.mean(m1, axis=d)
    m1 = np.mean(y.flatten())
    
    
    overall_mean = m1 
    HOW_MANY_STANDARD_DEVIATIONS = 2.
    #%just a scale of edges & centres value
    print 'using %d standard deviations, M=%d levels'%( HOW_MANY_STANDARD_DEVIATIONS, M)
    #%edges =   "values used for discretisation"
    #%centres2 =  "implied" value by each discretized level (used in dithering)"
    [edges, centres] = edges_centres(M, HOW_MANY_STANDARD_DEVIATIONS, overall_sigma, overall_mean)
    #%dinfo=centres
    if True:
        #if nargout == 2.:
        #%dinfo=centres
        dinfo = {}
        dinfo['edges'] = edges
        dinfo['centres'] = centres
        #% %dinfo = {'edges':edges,'centres':centres}
        #%dinfo = {edges, centres}
    
    ds = edges[2]-edges[1]
    z = regular_discretize(y, M, ds, edges[1])
    return z, dinfo
