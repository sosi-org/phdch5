import numpy as np
def myhist_int(vals, maxrange, startWith1=False):
    #print startWith1,"SW"
    #flag: startwith 1
    assert np.sum(np.abs(np.floor(vals)-vals))==0 #assert int
    assert float(int(maxrange)) == float(maxrange)
    #maxrange=int(M ** L +0.0001)
    edges = np.array(range(0,int(maxrange+0.0001)+1)) + 0.01

    #if startWith1:
    #    edges=edges+1
    if not startWith1:
        edges=edges-1
    #print edges

    #print vals.shape
    #assert len(vals.shape)==1
    assert np.max(vals.shape)==vals.size  #other dimensions are 1
    count,e2 = np.histogram(vals.flatten(), edges)

    assert count.shape == (maxrange,)
    #print sum(count), vals.size, startWith1
    assert sum(count)==vals.size

    return count