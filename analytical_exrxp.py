import numpy
def analytical_exrxp(tau_s_msec,sigma_s,tau_n_msec,sigma_n, fs_Hz,d_freq)
	#see analytical_lor.m
	#;
	# function [mi_persec, psdinfo]=analytical_lor(tau_s_msec,sigma_s,tau_n_msec,sigma_n, fs_Hz,d_freq)

	nyquist_freq = fs_Hz/2;
	# freq_arr =  - nyquist_freq : d_freq : nyquist_freq
	# np.exp(np.linspace(-2*np.pi, 2*np.pi, 100))
	numfreq = #d_freq
	freq_arr, freq_step =  np.linspace(-nyquist_freq, nyquist_freq, numfreq, False, True )

"""
MATLAB CODE:   analytical_lor.m


function [mi_persec, psdinfo]=analytical_lor(tau_s_msec,sigma_s,tau_n_msec,sigma_n, fs_Hz,d_freq)
%Analytical estimation of information in Lorentzian responses (i.e., entropy rate).
%Works accurately for small fs_Hz values.
%   tau_s_msec & tau_n_msec in milliseconds.
%   d_freq is optional (used for integration)
%   returns psdinfo (optional) containing PSDs for plotting.
%   note: psdinfo.freqs frequencies are the ones used in integration.
;
%see fft_lor.m

%set the default d_freq
if ~exist('d_freq','var')
    d_freq = 0.001235456789;
end
if numel(d_freq)==0
    error('usage error.');
end

nyquist_freq = fs_Hz/2;
freq_arr =  - nyquist_freq : d_freq : nyquist_freq ;
nyquist_omega = nyquist_freq*2*pi;
omega=freq_arr*2*pi;
taus=tau_s_msec/1000.0; %unit conversion
taun=tau_n_msec/1000.0;

%correcting the variance. (very accurate)
%The variance of the resulting signal would be tau/2
%which is the limit of the following: taus*2*atan(taus*nyquist_freq*2pi)/2pi
%in limit, it will be tau/2
% Mathematica code:
%  Limit[2*Integrate[1/(tau^-2 + omega^2)/(2*Pi), omega], omega -> Infinity]
%      which gives abs(tau)/2
%
% Mathematica code not using the lim
%  2*Integrate[1/(tau^-2 + omega^2)/(2*Pi), omega]
%    which is  (tau ArcTan[omega tau])/Pi
% Our MATLAB expression will be:
%   tau*atan(nyquist_freq*2*pi*tau))/pi
%    in limit it will be tau/2. in fact in MATLAB taus can be assigned Inf.
var_s = taus*atan(nyquist_omega*taus)/pi; %i.e., tau/2 if fs_Hz=Inf
var_n = taun*atan(nyquist_omega*taun)/pi;

%6april010 XM
%var_s=taus/2; % atan(taus*fs_Hz/2*2*pi)/pi*2
%var_n=taun/2; % atan(taun*fs_Hz/2*2*pi)/pi*2

psd_s=1./(omega.^2+taus.^(-2)) .* sigma_s.^2 / var_s;
psd_n=1./(omega.^2+taun.^(-2)) .* sigma_n.^2 / var_n;
psd_snr = psd_s ./ psd_n;

%for test:
%the following should be 1.0 (iof the vriance is successfully corrected)
ratio_s=sum(psd_s)*mean(diff(omega))/(2*pi)/sigma_s^2;
ratio_n=sum(psd_n)*mean(diff(omega))/(2*pi)/sigma_n^2;
%ratio_s-1  %very low error:    5.4888e-06
%ratio_n-1  % error:  1.1929e-05


if 0
    %%
figure(10);clf;
plot(freq_arr,psd_snr)
end

%integration
d_omega = mean(diff(omega));
mi_persec=0.5*sum(  log2(1+psd_snr )  ) * d_omega /(2*pi);

if nargout>1
    psdinfo=[];
    psdinfo.psd_s = psd_s;
    psdinfo.psd_n = psd_n;
    psdinfo.freqs = freq_arr; %only will contain only frequencies in nyquist range
    psdinfo.var_s = var_s;
    psdinfo.var_n = var_n;
end

"""