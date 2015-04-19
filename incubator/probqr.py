
import numpy as np
import scipy
import matcompat


from hx_types import is_any_int_type
from probrs import probrs
from probqrs import probqrs

def probqr(spk, nta, _range, q, f):

    # Local Variables: r, new_nt, f, s, ntr, M, L, q, pqq, _range, spk, prs, ns, nt
    # Function calls: probqrs, floor, max, sum, zeros, reshape, probqr, probrs, size
    #%it computes Pq(r) as the average of Pq(r|s)
    ntr = spk.shape[3-1] #matcompat.size(spk, 3.)
    ns = spk.shape[4-1] #matcompat.size(spk, 4.)
    L = spk.shape[2-1] #matcompat.size(spk, 2.)
    new_nt = nta/ f #int(np.floor(nta/ f)) #np.floor(matdiv(nt, f))
    assert is_any_int_type(nta[0])
    #M = 1 + np.max(spk.reshape([1, -1]))
    #M=1+max(reshape(spk,1,[]));
    M=int(1+max(spk.flatten()))
    #%spkt=squeeze(spk(1,:,_range,:));
    #%trials=(reshape(spkt,L,[]))';
    #pqq = np.zeros([1, M** L], int)
    pqq = np.zeros([1, M** L])
    for s in range(1, (ns)+1):
        assert type(s-1) is int
        assert s-1 >= 0
        assert new_nt[int(s)-1]<=_range.shape[1]
        assert int(s)-1>=0
        r = _range[int(s)-1,0:new_nt[int(s)-1]]
        prs = probrs(spk, r, s, M)
        #pqq = pqq+np.dot(probqrs(prs, L, q, M), new_nt[int(s)-1])
        pqq = pqq+probqrs(prs, L, q, M) * float(new_nt[int(s)-1])

    pqq = pqq / np.sum(new_nt)
    return pqq


"""
function pqq=probqr(spk,nt,range,q,f);
%it computes Pq(r) as the average of Pq(r|s)
ntr=size(spk,3);
ns=size(spk,4);
L=size(spk,2);
new_nt=floor(nt/f);
M=1+max(reshape(spk,1,[]));
%spkt=squeeze(spk(1,:,range,:));
%trials=(reshape(spkt,L,[]))';
pqq=zeros(1,M^L);

for s=1:ns
 r=range(s,1:new_nt(s));   
 prs=probrs(spk,r,s,M);  
 pqq=pqq+probqrs(prs,L,q,M)*new_nt(s);      
end
pqq=pqq/sum(new_nt);
"""