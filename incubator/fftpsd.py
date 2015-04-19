
import numpy as np
import scipy
import matcompat
import matplotlib.pylab as plt
from fftfreqs import fftfreqs

def fftpsd(signals, fs_Hz):

    # Local Variables: psd1, tr, fs_Hz, freqs, siglen, psd, signals, s, physical_length_sec, ff, ntrials
    # Function calls: sort, fftpsd, abs, fftfreqs, fft, length, warning, sprintf, size
    #%will return the power density: mV^2/Hz that is,
    #%      T*|f[n]|^2 in notation of SPIKES book 
    #%   or T*|c_n|^2 in my notation in my report  c_n=f[n]=f[-n]
    #%    or  |FFT[n]|^2 / N^2 * T in MATLAB's notation
    #% important equations:
    #%     c_n = FFT[n] / N.                     units = mV.
    #%     S(w) = T |c_n|^2 = power density.     units = mV^2/Hz.
    #%
    s = np.sort(matcompat.size(signals))
    #%only defined for signals = trials x bins
    if s[1] == 1:
        #%signals=signals';
        print 'Warning: size[0]=1; shape = %r'%(signals.shape)
    
    siglen = matcompat.size(signals, 2.)
    physical_length_sec = siglen / fs_Hz
    psd = 0.
    #ntrials = matcompat.size(signals, 1.)
    ntrials = signals.shape[0]
    for tr in np.arange(1., (ntrials)+1):
        ff = np.fft.fft(signals[int(tr)-1,:])
        #% median=22
        psd1 = (np.abs(ff) / siglen)**2.
        #%median: 5e-4
        psd = psd+psd1
        #%forgotten
        
    psd = psd / (1. * ntrials) * physical_length_sec
    #%PSD[w_n]= FFT[n]^2 /N^2 *T = FFT[n]^2 /N/N*(N/fs_Hz) = FFT[n]^2/N/fs_Hz
    freqs = fftfreqs(siglen, fs_Hz)

    return psd, freqs

    #%history:
    #% bug:    %forgotten psd=psd+psd1;
    #% bug:    psd1=abs(ff).^2 / length(siglen);  solved in 7 April 2010
    #% bug:    psd1=abs(ff).^2 / siglen;  solved in 7 April 2010
    #%xlabel('\omega_n');
    #%ylabel('S(\omega) = T*| c_n |^2 = T*|FFT[n]/N|^2');
    #return psd, freqs
    