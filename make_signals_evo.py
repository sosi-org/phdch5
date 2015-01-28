import pyentropy

def quantize_2d(z2d, M, uniform_code):
    #z2d_reshaped=z2d.reshape(z2d.size)
    #z2d_reshaped_q,bin_bounds, bin_centers = pyentropy.quantise(z2d_reshaped, M, uniform='bins')
    #z2d_q = z2d_reshaped_q.reshape(z2d.shape)
    z2d_reshaped=z2d.reshape(z2d.size)
    z2d_reshaped_q,bin_bounds, bin_centers = pyentropy.quantise(z2d_reshaped, M, uniform=uniform_code)
    z2d_q = z2d_reshaped_q.reshape(z2d.shape)
    return z2d_q

def sliding(zq, L, step=1): #also step between L elements?
    #zq: (nlen,ntr)
    assert step==1
    import numpy
    ns = int(numpy.floor((zq.shape[0]-L+1)/step)) #only tested for step==1
    #print "ns=%d"%ns
    ntr=zq.shape[1]
    #zL=numpy.zeros((ns,ntr), type(zq[0,0]))
    zL=numpy.zeros((L,ns*ntr), type(zq[0,0]))
    #nta = numpy.zeros(nlen, type(zq[0,0]))
    nta = numpy.zeros(ns,int)
    #nl = zq[
    start = 0
    ntrctr=0
    for i in range(ns):
        a=zq[0+start:L+start,:]  # Lx30
        zL[0:L,ntrctr+0:ntrctr+ntr] = a
        start = start + 1  #todo: start=i
        nta[i] = ntr
        ntrctr=ntrctr+ntr #todo: ntrctr = i*ntr
    return zL,nta

#import myshared
import exrxp
    
    
#main
#ntr=10 large info ntr=100  zero info!
ntr=100*10
nlen=1000


fs_Hz=1000.0 # Hz
tau_n_msec = 1.0 # msec
tau_s_msec = 5.0 # msec

#L=1 ---> Plateau.  L=2 ---> grows inf ly (bias?)


est_mi = []
est_M = []
est_a_mi = []

#import pyentropy
#for M in range(2,40,1):
#for M in range(2,80,4):
#for M in range(2,30,4):
#for M in [2,3,4,5,6,7,8,9,10,11]:
for M in [2,3,11]:

    #1.89216274871  M=10
    #0.243915598515 M=4

    import numpy

    #z2d = exrxp.exrxp_ntr (nlen,tau_msec/1000.0,fs_Hz, ntr)
    z2d = exrxp.exrxp_ntr (nlen,tau_n_msec/1000.0,fs_Hz, ntr) * 2 #*2*4
    z0 = exrxp.exrxp_ntr (nlen,tau_s_msec/1000.0,fs_Hz, 1)
    resp2d = z2d + numpy.tile(z0,[1,ntr])
    #print z2d.shape # nlen*ntr

    z2d_q = quantize_2d(resp2d, M, 'sampling') #'bins')
    z2dqL,nta = sliding(z2d_q, L=2)
    #z2dqL,nta = sliding(z2d_q, L=1)
    #import numpy
    from pyentropy import SortedDiscreteSystem
    s = SortedDiscreteSystem(z2dqL, (z2dqL.shape[0],M), len(nta), nta)
    s.calculate_entropies(method='qe', calc=['HX', 'HXY'])
    mi = s.I()
    print M
    print mi

    sigma_s=1
    sigma_n=1
    import analytical_exrxp as e
    a=e.exrxp_analytical_mi(tau_s_msec,sigma_s,tau_n_msec,sigma_n,fs_Hz)
    a_mi_persec = a[0]
    ami_perbin = a_mi_persec / fs_Hz
    #print "mi= %g "%( a_mi_persec / fs_Hz * nlen   ,)
    print "mi= %g "%( ami_perbin    ,)

    #error: it should be ttwice as large


    if 0:
        ts = exrxp.timesarr(nlen,fs_Hz)
        import matplotlib.pyplot
        p1=matplotlib.pyplot.plot(ts[0:50],resp2d[0:50,:])
        p1=matplotlib.pyplot.plot(ts[0:50],z0[0:50,0])
        #p1=matplotlib.pyplot.plot(ts,resp2d)
        #p1=matplotlib.pyplot.plot(ts,z0)
        #matplotlib.pyplot.axis([0, 0.3, -10,10]) #minx maxx, miny maxy
        matplotlib.pyplot.show()
        exit(0)

    est_mi.append(mi)
    est_M.append(M)
    est_a_mi.append(ami_perbin)


import matplotlib.pyplot as pp
p0=pp.plot(est_M[0],0.0)
p1=pp.plot(est_M,est_mi)
p2=pp.plot(est_M,est_a_mi)

legend_handles=[p1,p2]
labels=['est','analyt']
#ca=['r','g','b','k','m','c','y']
#pp.rcParams.update({'legend.fontsize': 8, 'legend.linewidth': 1})
pp.gca().set_xlabel('M')
pp.legend(labels)
pp.title("trials: %d"%(ntr,) )
pp.show()
    
        
# scp root@134.213.57.138:~/exhdd/exhdd/u/matlab/m9-markov5/makelorresponse.m ..
# rgrep analyti . |awk -F '%' {'print $1;'} |grep analyti|grep -v analytical\' |more
# analytical_lor.m
# scp root@134.213.57.138:~/exhdd/exhdd/u/matlab/m9-markov5/analytical_lor.m ..
