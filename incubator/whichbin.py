
import numpy as np
import scipy
import matcompat

import matplotlib.pylab as plt

def whichbin(val, edges):
    #todo: Pythonize
    # Local Variables: i, edges, e, val, idx
    # Function calls: warning, whichbin
    i = 1
    for e in edges:
        #%edges(k) <= x(i) < edges(k+1)
        if e>val:
            idx=i-1;

            if idx<1:
                print 'smaller than first interval'
            
            return idx
        i += 1

        
    if e == val:
        idx = i-1
    else:
        raise Exception('not found')
        
    
    return idx