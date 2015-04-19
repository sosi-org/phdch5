
import numpy as np
#import scipy
#import matcompat

# if available import pylab (from matlibplot)
import matplotlib.pylab as plt

def edges_centres(M, HOW_MANY_STANDARD_DEVIATIONS, sigma, mn):

    # Local Variables: ds_i, HOW_MANY_STANDARD_DEVIATIONS, centres2, mn, M, edges, edges0, del1, edges0_i, sigma
    # Function calls: length, edges_centres, Inf
    #%        ds=2*HOW_MANY_STANDARD_DEVIATIONS*sigma/M;
    #%        edges0=[-HOW_MANY_STANDARD_DEVIATIONS *sigma ...
    #%            : ds : ...
    #%            HOW_MANY_STANDARD_DEVIATIONS *sigma] ...
    #%            + mn;
    ds_i = 2*HOW_MANY_STANDARD_DEVIATIONS / float(M)
    edges0_i = np.arange(-HOW_MANY_STANDARD_DEVIATIONS, (HOW_MANY_STANDARD_DEVIATIONS)+(ds_i), ds_i)
    #edges0 = np.dot(edges0_i, sigma)+mn
    #print edges0_i.shape, M
    #print sigma.shape
    edges0 = (edges0_i * sigma)+mn
    edges = edges0
    #edges([1,end]) = [-Inf,Inf]; %first and last are replaced with Inf
    edges[0]=-np.Inf
    edges[len(edges)-1]=np.Inf
    #%first and last are replaced with Inf
    #%setting the levels (implied) values
    #%  edges : thresholds for cutting
    #%  centres2 : implied values
    #%
    if M == 2:
        #del1 = 0.
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
        #    centres2 = (centres2(2:end) + centres2(1:end-1))/2;
        centres2 = (centres2[1:]+centres2[0:0-1])/2.
        #%must contain 'M' levels
        
    
    return edges, centres2
