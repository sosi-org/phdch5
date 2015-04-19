
import numpy as np
import scipy
import matcompat
import matplotlib.pylab as plt

def fftfreqs(_len, fs_Hz):

    # Local Variables: physical_length, frq1, fs_Hz, FFT_freqs, _len, idx2
    # Function calls: fftfreqs, length
    physical_length = _len / float(fs_Hz)
    frq1 = np.dot(1./physical_length, np.arange(0, (_len-1)+1))
    FFT_freqs = frq1
    #idx2 = np.arange(length(FFT_freqs), (length(FFT_freqs)/2.+1.)+(-1.), -1.)
    #idx2 = np.arange(len(FFT_freqs), (len(FFT_freqs)/2.+1.)+(-1.), -1.)
    #idx2=length(FFT_freqs):(-1):length(FFT_freqs)/2+1;
    #    idx2 = range(len(FFT_freqs)-1, (len(FFT_freqs)/2+1-1-1) , -1 )  #one before last even in case of STEP=-1
    print "Warning: possibly wrong indices"
    idx2_1 = range(len(FFT_freqs)-1 -1, (len(FFT_freqs)/2+1-1-1)-1 , -1 )
    #FFT_freqs[int(idx2)-1] = -frq1[1:length(idx2)+1.]
    #    FFT_freqs(idx2) = -frq1(2:(length(idx2)+1));
    #FFT_freqs[int(idx2)-1] = -frq1[1:len(idx2)+1]
    #    print len(idx2)
    #    #FFT_freqs[idx2-1] = -frq1[1:len(idx2)+1]
    #    print len(frq1[1:(len(idx2)+1)])
    #    print len( FFT_freqs[idx2-1] )
    #print min(idx2),max(idx2)
    #FFT_freqs[idx2-1] = -frq1[1:(len(idx2)+1)]
    FFT_freqs[idx2_1] = -frq1[1:(len(idx2_1)+1)]
    
    #%    MATLAB's code:
    #% NFFT = 2^nextpow2(L); % Next power of 2 from length of y
    #% Y = fft(y,NFFT)/L;
    #% f = Fs/2*linspace(0,1,NFFT/2+1);
    return FFT_freqs