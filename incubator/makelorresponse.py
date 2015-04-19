
import numpy as np
import scipy
import math
#import matcompat
from make_lor9d import make_lor9d

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def makelorresponse(tau_s_msec, sigma_s, tau_n_msec, sigma_n, numtrials, siglen, fs_Hz_pre, dummy, downsample):

    # Local Variables: NORMALIZE, fft_SNR, tau_s_msec, fft_frqs_n, kerlen_s, a_psd_s, freqs, resp, data_info, nse, tau_n_msec, fftpsd, a_psd_n, fft_frqs_s, resp_info, nse_info, sig_info, noise, noises, sigma_s, mi_fft_persec, signals, sigma_n, fs2_Hz, fft_psd_n, sig_fft_trials, siglen_b, var_n, varinfo_s, normalizationinfo, psd, sig0, siglen, downsample, varinfo_n, numtrials, fft_psd_s, frqarr, fftinfo, timelen_sec, dummy, KERLEN_FACTOR, fft_s, r2, min_num_samples, SEC2MSEC, signal_s, kerlen_n, fftnumtr_s, fs_Hz_pre, signal1, fftnumtr_n, VNF_s0, num_samples, VNF_n0, var_s, signal, fft_n
    # Function calls: simulate_lorentzian7, warning, make_lor9d, repmat, tau_n, floor, trials, sum, sprintf, makelorresponse, tau_s, log2, max, SEC_MSEC, mod, std, double, fs_Hz, r, num2str, reshape, mean
    #% downsample = # of samples combined into one. i.e., b=8 means 8msec if fs=1KHz 
    #%fftinfo: not subject to downsampling
    NORMALIZE = True
    KERLEN_FACTOR = 5.
    #%4
    #%% Check the minumal length
    #%at least three times the correlations: max(taun, taus) *3
    #%min_num_samples=floor(max(tau_s_msec,tau_n_msec)/1000*3.0*fs_Hz); 
    min_num_samples = math.floor(max([tau_s_msec, tau_n_msec])/1000. * 3.0 * fs_Hz_pre)
    #%% signal generation
    #%frqarr_a=[-500:0.01:500];
    #%[signals,s_psd_a,s_psd_fft,s_fft_freqs]=simulate_lorentzian8(tau_s_msec/1000, numtrials, siglen, KERLEN_FACTOR , fs_Hz, frqarr_a);
    #%[noises, n_psd_a,n_psd_fft,n_fft_freqs]=simulate_lorentzian8(tau_n_msec/1000, numtrials, siglen, KERLEN_FACTOR , fs_Hz, frqarr_a);
    [signals, fft_s, fs2_Hz, varinfo_s] = make_lor9d((tau_s_msec/1000.), numtrials, siglen, KERLEN_FACTOR, fs_Hz_pre, downsample)
    [noises, fft_n, fs2_Hz, varinfo_n] = make_lor9d((tau_n_msec/1000.), numtrials, siglen, KERLEN_FACTOR, fs_Hz_pre, downsample)
    #%% manipulating the variances
    if NORMALIZE:
        VNF_s0 = 1./np.std(signals.flatten(1))
        VNF_n0 = 1./np.std(noises.flatten(1))
    else:
        VNF_s0 = 1.
        VNF_n0 = 1.
        
    
    signals = np.dot(np.dot(signals, VNF_s0), sigma_s)
    noises = np.dot(np.dot(noises, VNF_n0), sigma_n)
    #%was forgotten
    #fft_s : (psd_fft_psd,psd_fft_f)
    #fft_s.psd = np.dot(np.dot(fft_s.psd, VNF_s0**2.), sigma_s**2.)
    #fft_n.psd = np.dot(np.dot(fft_n.psd, VNF_n0**2.), sigma_n**2.)
    #in fact not used:
    fft_s_psd = np.dot(np.dot(fft_s[0], VNF_s0**2.), sigma_s**2.)
    fft_n_psd = np.dot(np.dot(fft_n[0], VNF_n0**2.), sigma_n**2.)
    #% sqrt(mean(var(signals,[],1)))  %precise
    #% sqrt(mean(var(signals,[],2)))  %not precise
    sig_info = 'tau_s = %g(ms) x %g'%( tau_s_msec, sigma_s)
    nse_info = 'tau_n = %g(ms) x %g'%( tau_n_msec, sigma_n)
    #%nse_info=['tau=',num2str(tau_n_msec),'(ms)x',num2str(sigma_n),''];
    #%signal1 = repmat(signals(1,:),[numtrials,1]); %why the variance is SOOOO small?
    #%r=signal1 +noises;
    signal1 = signals[0,:]
    #%The variance of signal1 can be different from the whole variance*  especially when the length is small
    #% see make_lor9.m
    #%r=repmat(signals(1,:),[numtrials,1]) + noises;
    #resp = matcompat.repmat(signals[0,:], np.array(np.hstack((numtrials, 1.))))+noises
    resp = np.tile(signals[0,:], (numtrials, 1))+noises
    #%already downsampled

    return resp

    #return resp, fftinfo, normalizationinfo

"""
    if 0.:
        #%% downsampling
        r2 = np.reshape(r, np.array(np.hstack((numtrials, matdiv(siglen, downsample), downsample))))
        resp = np.mean(r2, 3.)
        siglen_b = matdiv(siglen, downsample)
        
        #%todo: do downsampling on the reported fft (?)
        fftinfo = np.array([])
        fftinfo.fft_s = fft_s
        fftinfo.fft_n = fft_n
        normalizationinfo = np.array([])
        normalizationinfo.var_s = VNF_s0
        normalizationinfo.var_n = VNF_n0
    return []

    #%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SEC2MSEC = 1000.
    sig_fft_trials = trials
    return []
    #% Creates response adding signal (tau_s,sigma_s) and noise (tau_n,sigma_n).
    #% 'trials' is the number of response trials.
    #%
    #% uses 'sig_fft_trials' number of trials only to estimate signal's PSD based on fft.
    #% tau_s and tau_n are in milliseconds.
    #% siglen is in number of samples.
    #% todo: [r,mi]=makelorresponse2(10,1,10,1,100, 1000, 1000, 100)
    #%history:
    #%  2 Jan: min samples fixed. siginfo fixed.
    num_samples = siglen
    timelen_sec = matdiv(np.double(num_samples), fs_Hz)
    KERLEN_FACTOR = 4.
    NORMALIZE = 0.
    frqarr = np.array([])
    #%no analytical estimation needed/used here
    min_num_samples = np.floor(np.dot(matdiv(matcompat.max(tau_s, tau_n), SEC_MSEC)*3., fs_Hz))
    #%at least three times the correlations
    if siglen<min_num_samples:
        matcompat.warning('siglen=%d < min_num_samples=%d .\n', siglen, min_num_samples)
    
    
    fftnumtr_n = trials
    fftnumtr_s = sig_fft_trials
    kerlen_s = np.dot(tau_s, KERLEN_FACTOR)
    [signal_s, a_psd_s, fft_psd_s, fft_frqs_s] = simulate_lorentzian7(matdiv(tau_s, SEC_MSEC), fftnumtr_s, num_samples, kerlen_s, fs_Hz, frqarr)
    signal_s = np.dot(signal_s, sigma_s)
    if NORMALIZE:
        VNF_s0 = matdiv(sigma_s, np.std(signal_s.flatten(1)))
    else:
        VNF_s0 = 1.
        
    
    signal_s = np.dot(signal_s, VNF_s0)
    sig_info = np.array(np.hstack(('tau=', num2str(tau_s), '(ms)x', num2str(sigma_s), \')))
    kerlen_n = np.dot(tau_n, KERLEN_FACTOR)
    #%[tau_n,fftnumtr_n, num_samples, kerlen_n, fs_Hz]
    [nse, a_psd_n, fft_psd_n, fft_frqs_n] = simulate_lorentzian7(matdiv(tau_n, SEC_MSEC), fftnumtr_n, num_samples, kerlen_n, fs_Hz, frqarr)
    nse = np.dot(nse, sigma_n)
    if NORMALIZE:
        VNF_n0 = matdiv(sigma_n, np.std(nse.flatten(1)))
    else:
        VNF_n0 = 1.
        
    
    nse = np.dot(nse, VNF_s0)
    nse_info = np.array(np.hstack(('tau=', num2str(tau_n), '(ms)x', num2str(sigma_n), \')))
    #% creating thedata_info response
    sig0 = signal_s[0,:]
    resp = matcompat.repmat(sig0, np.array(np.hstack((trials, 1.))))+nse
    #% carrier signal (whole signal)
    resp_info = np.array(np.hstack(('signal: ', sig_info, ', noise:', nse_info)))
    #%if exist('data_info')
    data_info = resp_info
    #%end
    #%if exist('fftpsd')
    fftpsd = np.array([])
    fftpsd.signal = np.dot(np.dot(fft_psd_s, sigma_s**2.), VNF_s0**2.)
    fftpsd.noise = np.dot(np.dot(fft_psd_n, sigma_n**2.), VNF_n0**2.)
    fftpsd.freqs = fft_frqs_s
    #%same as fft_frqs_n
    #%end
    fft_SNR = np.dot(np.dot(fft_psd_s, sigma_s**2.), VNF_s0**2.)/np.dot(np.dot(fft_psd_n, sigma_n**2.), VNF_n0**2.)
    mi_fft_persec = matdiv(np.dot(0.5, np.sum(np.log2((1.+fft_SNR)))), timelen_sec)
    return resp, fftinfo, normalizationinfo
"""