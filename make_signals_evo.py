import pyentropy

#Two formats for the sample: TxN and LSN:
#   LxSN:  Lx(S*Tr) = LST = L x Stim x Ntr = LxSxN = LSN
#   Txn:   LT = LenTr = Time x Ntr = TxN = TN
#SignalsSample_LT:

def quantize_TxN(z_TxN, M, uniform_code):
    #z2d_reshaped=z2d.reshape(z2d.size)
    #z2d_reshaped_q,bin_bounds, bin_centers = pyentropy.quantise(z2d_reshaped, M, uniform='bins')
    #z2d_q = z2d_reshaped_q.reshape(z2d.shape)
    #z2d: # [nlen x ntr]
    z_NxT=z_TxN.transpose()
    z_NxT_shape=z_NxT.shape
    #print z2d[0:10,0] # [ntr  x nlen ]
    z_Arr=z_NxT.reshape([z_NxT.size]) #straighted.  z2d.size=total elements
    #print z2d_reshaped[0:10] #if not trsnsposed,  tr1,tr2,tr3,  tr1,tr2,tr3

    zq_Arr,bin_bounds, bin_centers = pyentropy.quantise(z_Arr, M, uniform=uniform_code)
    zq_NxT = zq_Arr.reshape(z_NxT_shape)
    return zq_NxT.transpose()

def sliding(zq_TxN, L, step=1, offset=0): #also step between L elements?
    """ converts a TxN into LxSN """
    #zq: (nlen,ntr) = (T,N)
    #assert step==1
    import numpy,math
    ns = int(math.floor((zq_TxN.shape[0]-L+1)/step)) #only tested for step==1
    #print "ns=%d"%ns
    ntr=zq_TxN.shape[1]
    #zL=numpy.zeros((ns,ntr), type(zq[0,0]))
    z_LxSN=numpy.zeros((L,ns*ntr), type(zq_TxN[0,0]))
    #nta = numpy.zeros(nlen, type(zq[0,0]))
    nta = numpy.zeros(ns,int)
    #nl = zq[
    word_start = offset #argument: offset
    SN_ctr = 0 #The SN dimension is (Stim x Ntr), i.e, arrays of ntr repeated Stim (=ns) times.  Lx SN=Lx(Stim x Ntr)
    for i in range(ns):
        a=zq_TxN[0+word_start:L+word_start,:]  # Lx30
        z_LxSN[0:L,SN_ctr+0:SN_ctr+ntr] = a
        #start = start + 1   #todo: start=i
        word_start = word_start + step  #todo: start=i*step
        nta[i] = ntr
        SN_ctr=SN_ctr+ntr #todo: ntrctr = i*ntr
    return z_LxSN,nta

#import myshared
import exrxp
    
    
#main
#ntr=10 large info ntr=100  zero info!
ntr=100*10
nlen=1000

#ntr=2;nlen=3

fs_Hz=1000.0 # Hz
tau_n_msec = 1.0*100 # msec
tau_s_msec = 5.0*100 # msec

#test
tau_n_msec = 1.0*100/10 # msec
tau_s_msec = 1.0*100/10 # msec


# sigma_s=10;sigma_n=2 #why faster growing by M ?
sigma_s=1
sigma_n=1

#L=1 ---> Plateau.  L=2 ---> grows inf ly (bias?)

L=2

est_mi = []
est_M = []
est_a_mi = []

#import pyentropy
#for M in range(2,40,1):
#for M in range(2,80,4):
#for M in range(2,30,4):
for M in [2,3,4,5,6,7,8,9,10,11,15,20,30,50]:
#for M in [2,3,11]:

    #1.89216274871  M=10
    #0.243915598515 M=4

    import numpy

    #z2d = exrxp.exrxp_ntr (nlen,tau_msec/1000.0,fs_Hz, ntr)
    z2d = exrxp.exrxp_ntr (nlen,tau_n_msec/1000.0,fs_Hz, ntr) * sigma_n #* 2 #*2*4
    z0 = exrxp.exrxp_ntr (nlen,tau_s_msec/1000.0,fs_Hz, 1) * sigma_s
    resp2d = z2d + numpy.tile(z0,[1,ntr])
    #print z2d.shape # nlen*ntr

    z2d_q = quantize_TxN(resp2d, M, 'sampling') #'bins')
    #z2dqL,nta = sliding(z2d_q, L=2)
    #z2dqL,nta = sliding(z2d_q, L=1)
    z2dqL,nta = sliding(z2d_q, L=L)

    #import numpy
    from pyentropy import SortedDiscreteSystem
    s = SortedDiscreteSystem(z2dqL, (z2dqL.shape[0],M), len(nta), nta)
    s.calculate_entropies(method='qe', calc=['HX', 'HXY'])
    mi = s.I() / L
    print M
    print mi

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
import myshared
p0=pp.plot(est_M[0],0.0)
p1=pp.plot(est_M,est_mi)
p2=pp.plot(est_M,est_a_mi)

legend_handles=[p1,p2]
labels=['est','analyt']
#ca=['r','g','b','k','m','c','y']
#pp.rcParams.update({'legend.fontsize': 8, 'legend.linewidth': 1})
pp.gca().set_xlabel('M')
pp.legend(labels, loc=myshared.LEGEND_CONST.lower_right)
pp.title("trials: %d"%(ntr,) )
pp.show()
    
        
# scp root@134.213.57.138:~/exhdd/exhdd/u/matlab/m9-markov5/makelorresponse.m ..
# rgrep analyti . |awk -F '%' {'print $1;'} |grep analyti|grep -v analytical\' |more
# analytical_lor.m
# scp root@134.213.57.138:~/exhdd/exhdd/u/matlab/m9-markov5/analytical_lor.m ..



import numpy.fft
#def fft_psd(signal,fs_Hz):
#    a=numpy.fft.fft()
#    pass


#def fft_freqs(l,fs_Hz):
#    physical_length = l / fs_Hz;
#    frq1= 1/physical_length * [ 0:(len-1) ]
#    FFT_freqs = frq1
#    idx2=length(FFT_freqs):(-1):length(FFT_freqs)/2+1
#    FFT_freqs(idx2) = -frq1(2:(length(idx2)+1))

#cd ~/exhdd/exhdd/u/matlab/m9-markov5/
#cat ./m7-Markov_proj3/fftfreqs.m
#function FFT_freqs=fftfreqs(len,fs_Hz)
#    physical_length = len / fs_Hz;
#    frq1= 1/physical_length * [ 0:(len-1) ];
#    FFT_freqs = frq1;
#    idx2=length(FFT_freqs):(-1):length(FFT_freqs)/2+1;
#    FFT_freqs(idx2) = -frq1(2:(length(idx2)+1));
#end
    
#%    MATLAB's code:
#% NFFT = 2^nextpow2(L); % Next power of 2 from length of y
#% Y = fft(y,NFFT)/L;
#% f = Fs/2*linspace(0,1,NFFT/2+1);
