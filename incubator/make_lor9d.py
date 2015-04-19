
import numpy as np
import scipy
import matcompat
from fftpsd import fftpsd
import matplotlib.pylab as plt

def make_lor9d(tau_sec, numtrials, siglen_pre, kerlen_ratio, fs_Hz_pre, downsample):

    # Local Variables: siglen_pre, xlim, kerlen_ratio, psd_from_fft, psd_fft, sc_d_0, sc_rep, downsample, s2, tt, sc_d, var, kerlen, tau_sec, df, i0, c, psd, varinfo, numtrials, name, fs_Hz_post, siglen_post, VERBOSE, f, i, signal, freqs, kern, fs_Hz_pre, s, sc, x
    # Function calls: autocorrel, figure, warning, fprintf, make_lor9d, diff, find, size, plot, randn, conv, floor, reshape, sum, nargout, zeros, length, pi, sort, fftpsd, atan, squeeze, mod, decimate, exp, clf, mean
    #%9d: downsamples as well
    #%
    #%tau_sec in seconds. (can be big)
    #%siglen in samples.
    #%kerlen_ratio is about 2,3,4,...
    #%
    #%returns:
    #%    signal: trials x siglen
    #%    psd and varinfo are generated only if demanded.
    #%    only if varinfo is mentioned, the 8 variances are printed.
    #%main method imported from simulate_lorentzian8.m
    #%known problems:
    #%   1-bad at low fs_Hz (?)
    #%   2- bad at low tau_sec
    #%   But ok for natural ranges (e.g., A1 cortex data)
    VERBOSE = True
    #%for debuging
    if VERBOSE:
        print 'tau=%g(ms), trials=%d, len=%d (before downsampling), ker x=%g'%(tau_sec*1000., numtrials, siglen_pre, kerlen_ratio)
    
    
    if tau_sec > siglen_pre / fs_Hz_pre:
        matcompat.warning('tau_sec > T')
        #% although the result will be correct
    
    
    #%making the kernel
    kerlen = np.dot(kerlen_ratio, tau_sec)
    tt = np.arange(0., (kerlen)+(1./fs_Hz_pre), 1./fs_Hz_pre)
    kern = np.exp((-tt / tau_sec))
    if VERBOSE:
        #%this should be close to 1.0 :
        print('%g ?= 1.0\n'%( ((np.sum(kern) / tau_sec) / fs_Hz_pre)))
    
    #%pre = before downsampling
    #%post = after downsampling
    siglen_post = (siglen_pre / downsample)
    #%making the signal.
    signal = np.zeros([numtrials, siglen_post])
    for i in np.arange(1., (numtrials)+1):
        #%notes in simulate_lorentzian6.m



        #%notes in simulate_lorentzian6.m
        #s = plt.randn(1., (siglen_pre+len(kern)*2.+4.))
        s = plt.randn( (siglen_pre+len(kern)*2.+4.))
        #print s.shape
        #%+2!!
        xlim = len(kern)+np.array(np.hstack((1., siglen_pre)))-1.+1.+2.
        #%throwing away a few extra samples (for very long correlation lengths)
        #s2 = np.dot(plt.conv(s, matdiv(kern, fs_Hz_pre)), fs_Hz_pre)
        #print s.shape
        #print (kern / fs_Hz_pre).shape
        s2 = np.convolve(s, kern / fs_Hz_pre)* fs_Hz_pre
        #%integration
        sc = s2[int(xlim[0])-1:xlim[1]]
        #%cut the useful portion
        #%signal(i,:) = sc; %s2
        #%downsampling
        #%assert(mod(siglen_pre,downsample)==0);
        #%if 1
        #%sc_rep=reshape(sc,[1,siglen_pre/downsample,downsample]); %1 trial
        #sc_rep = np.reshape(sc, np.array(np.hstack((1., downsample, matdiv(siglen_pre, downsample)))))
        sc_rep = np.reshape(sc, [1, downsample, siglen_pre/ downsample])
        #%1 trial
        #%assert(floor(siglen_pre/downsample)==siglen_post)
        #%sc_d_0=mean(sc_rep,3);
        sc_d_0 = np.mean(sc_rep, axis=2-1)
        #%siglen_b=siglen/downsample;
        #%end

        #print signal.shape
        #print i,

        #sc_d = matcompat.decimate(sc, downsample)
        sc_d = sc[0::downsample]
        #signal[int(i)-1,:] = sc_d
        #print sc_d.shape

        signal[int(i)-1][:] = sc_d
        

        #%s2
        #%A=0
        #%if A
        if False:
            plt.figure(1.)
            plt.clf
            plt.plot((np.arange(1., (siglen_post)+1)-1.), np.squeeze(sc_d_0))
            plt.hold(on)
            plt.plot((np.arange(1., (siglen_post)+1)-1.), sc_d, 'r')
            #%plot((1:length(sc))/4/5+1,squeeze(sc),'k')
            plt.plot(matdiv(np.arange(1., (siglen_pre)+1)-1., downsample), np.squeeze(sc), 'k')
            #%end
        #%end

        
    #fs_Hz_post = matdiv(fs_Hz_pre, downsample)
    fs_Hz_post = fs_Hz_pre / downsample
    #%making & testing the fft
    #%FFT is done on DOWNSAMPLED signals (after downsampling)
    #% Only if fft psd is demanded. If not, only the signal is generated

    nargout = 1000 #extra_output +1

    if nargout > 1.:
        [psd_from_fft, freqs] = fftpsd(signal, fs_Hz_post)
        #psd_fft = np.array([])
        psd_fft_psd = psd_from_fft
        psd_fft_f = freqs
        psd_fft = (psd_fft_psd,psd_fft_f)
        if nargout > 2:
            varinfo = [{},{},{},{},{},{},{}] #np.array([])
            #%there are eight (!!) variance measures that can be used for normalization!!
            #%please note that the variance rate (i.e., var(:)/fs_Hz_post) is multiplied by fs_Hz_post.
            #% to get the approximation for var(signal(:))
            df = np.mean(np.diff(np.sort(freqs)))
            #%=1.0!
            varinfo[0]['_name'] = 'fft'
            varinfo[0]['_var'] = ((np.dot(np.sum(psd_from_fft), df) / fs_Hz_post)* fs_Hz_post)
            varinfo[1]['_name'] = 'vertical variance'
            #%var_v; % var_1;%varinfo(2)['_name']='mean(var(dim1))';
            #varinfo(2).var = mean(var(signal,[0],1))/fs_Hz_post * fs_Hz_post;
            varinfo[1]['_var'] = np.mean(np.var(signal,axis=0).flatten()) / fs_Hz_post* fs_Hz_post
            varinfo[2]['_name'] = 'var(:)'
            #%var_2;
            #varinfo(3).var = var(signal(:))/fs_Hz_post * fs_Hz_post;
            varinfo[2]['_var'] =  np.var(signal.flatten()) / fs_Hz_post * fs_Hz_post

            varinfo[3]['_name'] = 'horizontal variance'
            #%var_h;
            #varinfo(4).var = mean(var(signal,[0],2))/fs_Hz_post * fs_Hz_post;
            varinfo[3]['_var'] = np.mean(np.var(signal,axis=2-1).flatten()) / fs_Hz_post * fs_Hz_post

            #%can be very low if tau_sec is very big compared to siglen/fs_Hz
            varinfo[4]['_name'] = 'analytical varinace (nyquist)'
            #%var_a;
            #varinfo(5).var = 2*tau_sec*atan(2*pi*fs_Hz_post*tau_sec)/(2*pi) * fs_Hz_post;
            varinfo[4]['_var'] = 2.*tau_sec * np.arctan(2.*np.pi * fs_Hz_post * tau_sec) /  (2.*np.pi) * fs_Hz_post
            varinfo[5]['_name'] = 'ideal variance'
            #%var_a_inf;
            varinfo[5]['_var'] = tau_sec/2. * fs_Hz_post
            varinfo[6]['_name'] = 'kernel norm'
            varinfo[6]['_var'] = np.sum(kern) / tau_sec / fs_Hz_post
            #%this is not a 'variance'. This should be 1.0.
            #%slow
            if False:
                #autocorrel not converted into python
                [x, c] = autocorrel(signal, fs_Hz_post)
                i0 = nonzero((x >= 0.), 1., 'first') #find(,'first')
                print "Warning: nonzero()"
                #%plot(x*1000,c)
                varinfo[7]['_name'] = 'autocorrel(0) var'
                #varinfo(8).var=c(i0) * fs_Hz_post;
                varinfo[7]['_var'] = np.dot(c[int(i0)-1], fs_Hz_post)
                #%not very reliable.

            
            
            #if VERBOSE:
            #    fprintf('var*fs=')
            #    fprintf(' %g, ', np.array(np.hstack((varinfo[:]['_var']))))
            #    fprintf('.\n')

            return signal, psd_fft, fs_Hz_post, varinfo
            
    #return signal, psd_fft, fs_Hz_post
            
    #return signal #, psd_fft, fs_Hz_post, varinfo
    raise Exception("Bad usage")