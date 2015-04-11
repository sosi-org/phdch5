import numpy as np
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
    assert min(nta)/f>0, "Too few trials (%d) divided by %r"%(min(nta),f)
    #del nta
    m=max(new_nta) #m=max(new_nt);
    #r=zeros(ns,m);
    r=np.zeros([ns,m],int) #todo: int -> ?
    #for s=1:ns
    for s in range(ns):
        #r(s,1:new_nt(s))=range1(s,1+(k-1)*new_nt(s):k*new_nt(s));
        #print new_nta[s],"=",k
        r[s,0:new_nta[s]] = range1[s, range((k-1)*new_nta[s], k*new_nta[s])  ] #todo: not tested
    #end
    return r
