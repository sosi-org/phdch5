import numpy as np
import exrxp
import pyentropy

EPS=0.000000001

def quantize_TxN(z_TxN, M, uniform_code):
    z_NxT=z_TxN.transpose()
    z_NxT_shape=z_NxT.shape
    z_Arr=z_NxT.reshape([z_NxT.size]) #straighted.  z2d.size=total elements
    zq_Arr,bin_bounds, bin_centers = pyentropy.quantise(z_Arr, M, uniform=uniform_code)
    zq_NxT = zq_Arr.reshape(z_NxT_shape)
    return zq_NxT.transpose()

def sliding(zq_TxN, L, step=1, offset=0): #also step between L elements?
    """ converts a TxN into LxSN """
    import numpy,math
    ns = int(math.floor((zq_TxN.shape[0]-L+1)/step)) #only tested for step==1
    ntr=zq_TxN.shape[1]
    z_LxSN=numpy.zeros((L,ns*ntr), type(zq_TxN[0,0]))
    nta = numpy.zeros(ns,int)
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

def downsample(z2d, downsample_b):
    print "todo"
    return z2d

def _isint(x):
    #return type(shape[0]) is int
    return issubclass(type(x), np.integer) or issubclass(type(x), int)
    #n.dtype('int8').kind == 'i'


def makelorresponse(tausigma_s, tausigma_n, fs_Hz, shape, downsample_b):
    #def makelorresponse(tausigma_s, tausigma_n, fs_Hz, shape, what=[], downsample_b):
    #def makelorresponse(tau_s,sigma_s, tau_n, sigma_n, ntr, nlen, fs_Hz,what=[], downsample_b):
    #nlen = siglen_sec*fs_Hz
    (nlen,ntr) = shape
    assert _isint(shape[0]), str(type(shape[0]))
    assert _isint(shape[1]), str(type(shape[1]))

    assert type(downsample_b) is int
    assert downsample_b > 0
    print "1"
    z0 = exrxp.exrxp_ntr (nlen,tausigma_s[0],fs_Hz, 1) * tausigma_s[1]
    print "2:",ntr, "x",nlen
    return
    z2d = exrxp.exrxp_ntr (nlen,tausigma_n[0],fs_Hz, ntr) * tausigma_n[1] #* 2 #*2*4
    print "3"
    resp2d = z2d + np.tile(z0,[1,ntr])
    return downsample(resp2d,downsample_b)

def zzz(resp2d,L,M):
    z2d_q = quantize_TxN(resp2d, M, 'sampling') #'bins')
    z2dqL,nta = sliding(z2d_q, L=L)
    return z2dqL,nta

def discr(raw,M):
    print "error"
    return resp_dig, dithinfo
def dith_unsure(raw,centres):
    print "error"
    return resp_dith, dithinfo

def almost_eq(a,b):
    return abs(a-b)<EPS

def make_signals(taustd_s,taustd_n, M):
    #if 0
    #%4c1: failed (too slow)
    #% from test_lor3c2
    #% 1) 200 samples only.
    #% 2) down sampling changes SNR
    #% 3) different length for different L s

    ##%tau_s=25/4;
    ##%tau_s=25;
    #tau_s=25.0/1
    #std_s=3.53 #%*200;
    #tau_n=12.0/1
    #std_n=5.55 #%*200;



    #%NTA=2.^[4:5];
    #%NTA=2.^[4:8,9];
    NTA=np.power(2,[4,5,6,8,10,12])
    NTA=np.power(2,[4,5])
    #nt_generate=max(NTA);
    print NTA

    #fs_Hz=1000
    binlen_sec=7.0/1000.0 #7,3,5,1
    siglen_sec=1.000 #quick test
    downsample_b = 7
    if False:
        #siglen_sec=10.000;
        print siglen_sec, siglen_sec % binlen_sec
        siglen_sec=np.floor(siglen_sec/binlen_sec)*binlen_sec
        print siglen_sec, siglen_sec % binlen_sec
        downsample_b=binlen_sec*fs_Hz #%1, fs=1000 ===> 1   2==>2
        nlen = siglen_sec*fs_Hz
        #problem: print  0.994000 % 0.007000 - 0.007
        #-2.60208521397e-17
        #np.float     np.float128  np.float16   np.float32   np.float64   np.float_    np.floating
        #
        #%downsample_b happens to be equal to b, if fs=1000
        assert almost_eq(siglen_sec % binlen_sec, 0), "%f %% %f = %f"%(siglen_sec, binlen_sec, siglen_sec % binlen_sec,)

    nlen = int(np.floor(siglen_sec/binlen_sec))
    print nlen,siglen_sec/binlen_sec
    fs_Hz = 1.0/(binlen_sec/float(downsample_b))
    print fs_Hz
    print "%f"%(fs_Hz - int(fs_Hz),)

    print "raw",nlen,max(NTA),
    #r2_raw = makelorresponse(tau_s,std_s, tau_n, std_n, max(NTA), nlen, fs_Hz,[], downsample_b)
    #r2_raw = makelorresponse(taustd_s, taustd_n, fs_Hz, (nlen,max(NTA)), [], downsample_b)
    r2_raw = makelorresponse(taustd_s, taustd_n, fs_Hz, (nlen,max(NTA)), downsample_b)
    print "war"

    siglen_bins=siglen_sec/binlen_sec
    del siglen_sec
    assert siglen_bins == r2_raw.shape[0]

    return r2_raw





BIAS_TYPE=0

taustd_s=(25.0/1, 3.53)
taustd_n=(12.0/1, 5.55)
M=5  #4,8,5,3,15

print "***-4"
r2_raw =make_signals(taustd_s,taustd_n, M)
print "***-3"
resp_dig,dithinfo1 = discr(r2_raw, M) #%not dithered
print "***-2"
resp_dith,dithinfo2 = dith_unsure(r2_raw, dithinfo1.centres) #%not dithered

print "***-1"

resp_dig=resp_dith
del  dithinfo
resp_dig=resp_dig-1

print "***1"
resp_cont=r2_raw;
print "***2"
resp_cont=resp_cont-resp_cont.flatten().mean()
print "***3"
resp_cont=resp_cont/resp_cont.flatten().std()/2.0*1*float(M)/float(M-1)
print "***4"
#% [-1,+1]
resp_cont=(resp_cont+1.0)/2.0;  #% [0,1]
resp_cont=resp_cont*float(M-1);
if True:
    print resp_cont.reshape
    exit(0)
    """
    resp_cont(resp_cont(:)<0)=0;
    resp_cont(resp_cont(:)>M-1)=M-1;
    """

"""
#%clear r2_raw;


#% analytical formula
#%fs_Hz is used for Nyquist frequency
#%ami=analytical_lor(tau_s,std_s, tau_n, std_n, fs_Hz, 0.0001);
#%ami_perbin=mi/fs_Hz*binlen_msec;
#
#% %WHY?!
#%a=analytical_lor(tau_s/binlen_msec,std_s, tau_n/binlen_msec, std_n, fs_Hz/binlen_msec, 0.0001);
#%ami_perbin=a / (fs_Hz/binlen_msec);
#% (tau_s_msec,  sigma_s,  tau_n_msec,  sigma_n, fs_Hz, d_freq)
#%Ia_bits_per_sec
#% Ia_bps --> ami_bps = analytical mi
ami_bps=analytical_lor(tau_s,std_s, tau_n, std_n, 1/(binlen_msec/1000), 0.0001);
#%ami_perbin=ami_bps* (binlen_msec/1000);
#% 1/ (fs_Hz/binlen_msec) = binlen_msec / fs = above (bcoz, fs=1000)

#% previous: 0.2362
#% new: 2.4512e-04

binlen_sec = (binlen_msec/1000); %used later for normalization



clear a;
clear binlen_msec;



clearvars -except   resp_cont resp_dig M NTA binlen_sec siglen_bins tau_s tau_n ami_bps ...
    r2_raw std_* methid;



#%size(respcut_dig)
if 0
    %%
    %mean(mean(resp_dig,1))
    [c,d]=xcorr((mean(resp_dig,1)-1-(M-1)/2+1)/M*2,'unbiased');
    figure(10);clf;
    % / sqrt(std_s^2+std_n^2)
    h=plot(d*binlen_sec*1000,c /std_s^2,'k.-');
    set(h,'linewidth',2);
    set(h,'displayname','signal');
    
    xlim([-1,1]*100*1);
    %set(gca,'xtick',-200:10:200);
    title(sprintf('\\tau_s, \\tau_n = %g,%g [msec]',tau_s,tau_n));
    
    nn=size(resp_dig,1);
    c1=0;
    for tri=1:nn
        %mean(resp_dig(tri,:))
        [c,d]=xcorr((resp_dig(tri,:)-1-(M-1)/2+1)/M*2,'unbiased');
        c1=c1+c/nn;
    end
    clear nn;
    hold on;
    h=plot(d*binlen_sec*1000,c1 /std_n^2,'r.-');
    set(h,'linewidth',2);
    set(h,'displayname','noise');
    
    tt=d*binlen_sec;
    h=plot(tt*1000,binlen_sec*exp(-abs(tt)/(tau_s/1000)),'k--');
    set(h,'displayname','signal analytical');
    h=plot(tt*1000,binlen_sec*exp(-abs(tt)/(tau_n/1000)),'r--');
    set(h,'displayname','noise analytical');
    h=legend('show');
    set(h, 'box','off');
    set(h, 'location','NorthWest');
    xlabel('Time [msec]');
    
    title('cross correlation');
    % my_save_png('test_lor4f1---signal-xcorr-v12-temp',{},[4,3]);
    
end

begintime=now;
tic
%estims=cell(4,4);
%LARR=[1,4,8,10];
%LARR=[1,2:2:10];
%LARR=[1,2:2:8];
%LARR=[1,2,3,6];
%LARR=[1,2,3,6,7];
%LARR=[1]; %
LARR=[1,3,6,7];
MAX_L_DIRECT = Inf;

%for nti=10:length(NTA)
for nti=1:length(NTA)
    nt=NTA(nti);
    for li=1:length(LARR)
        L=LARR(li);
        %lenL=floor(siglen_msec/binlen_msec/L)*L;
        %lenL=floor(siglen_msec/L/binlen_msec)*L*binlen_msec;
        lenL=floor(siglen_bins/L)*L;
        
        %respcut=resp(:,1:lenL)';
        %  respcut=resp(1:nt,1:lenL)';
        if 0 %if dithering
            respcut_cont=resp_cont(1:nt,1:lenL)';
        end
        respcut_dig=resp_dig(1:nt,1:lenL)';
        % len x trials
        %spk0 = zeros(1,LNum,size(signal,1), newlen );
        %spk=reshape(respcut,  [1,L,size(resp)]);
        % spk__=reshape(respcut,  1,L,lenL/L,[]);
        if 0 %if dithering
            spk__cont=reshape(respcut_cont,  1,L,lenL/L,[]);        % 1 x L x len x trials
            spk2_cont=permute(spk__cont,[1,2,4,3]); clear spk__c;%slow
        end
        
        % 1 x L x trials x len
        
        
        spk__dig=reshape(respcut_dig,  1,L,lenL/L,[]);
        spk2_dig=permute(spk__dig,[1,2,4,3]); clear spk__dig; %slow
        
        nts=ones(1,lenL/L)*nt;
        
        fprintf('spk2: %dx%dx%dx%d [',size(spk2_dig));
        
        if  L<=MAX_L_DIRECT
            
            ns=lenL/L;
            nta=(1:ns)*0 + nt;
            
            
            if 0 %dithering
                
                %[prs3,ps3]=my_prs_dith1(spk2, nta);
                %   [prs3_dith,ps3]=my_prs_dith1(spk2_c-1, nta);
                %[prs3_dith,ps3]=my_prs_dith1(spk2_cont-0, nta);
                [prs_dith,ps_dith]=my_prs_dith1(spk2_cont-0, nta);
                
                %    [prs3_dig,ps3]=my_prs(spk2-1, nta);
                %    sum(abs(prs3_dith(:)-prs3_dig(:)))
                
                %   if 0
                %       %[prs1,ps1]=my_prs(spk2, nta);
                %       [prs1,ps1]=my_prs(spk2_dig, nta);
                %   end
                
                [prs_dig,ps_dig]=my_prs(spk2_dig, nta);
                
                %prs=prs3_dith;
                %ps=ps3;
                fprintf('p');
                
                
                hrs0_th=my_hrs(prs_dith,ps_dith, methid);
                pr_dith=my_pr(prs_dith,ps_dith);
                hr0_th=my_hr(pr_dith, methid);
                
                
                %hr1=hr(spk2_dig,nts,methid);
                %hrs1=hrs(spk2_dig,nts,methid);
                
                hrs_dig=my_hrs(prs_dig,ps_dig, methid);
                pr_dig=my_pr(prs_dig,ps_dig);
                hr_dig=my_hr(pr_dig, methid);
                
                %[ my_hr(ps_dith, methid),my_hr(ps_dig, methid)]
                m22= [...
                    my_hr(pr_dith, methid),my_hr(pr_dig, methid) ...
                    ; ...
                    my_hrs(prs_dith,ps_dith, methid),my_hrs(prs_dig,ps_dig, methid)];
                
                m22
                
                %myminmax(spk2_dig(:))  % 1--4
                %myminmax(spk2_cont(:)) % 0--3
                
                estims{li,nti}.hr_d= hr_dig; %hr1;
                estims{li,nti}.hrs_d=hrs_dig; %hrs1;
                
                estims{li,nti}.hr_th=hr0_th;
                estims{li,nti}.hrs_th=hrs0_th;
            end
            
            
            hr1_dig=hr(spk2_dig,nts,methid);
            hrs1_dig=hrs(spk2_dig,nts,methid);
            
            estims{li,nti}.hr_d= hr1_dig; %hr1;
            estims{li,nti}.hrs_d=hrs1_dig; %hrs1;
            if 1 %no dithering
                estims{li,nti}.hrs_sh=hrs_shuff(spk2_dig,nts,methid);
                estims{li,nti}.hrs_ind=hrsind(spk2_dig,nts,methid);
                estims{li,nti}.xi=xi(spk2_dig,nts,methid);
            end
        else
            estims{li,nti}.hr=0;
            estims{li,nti}.hrs=0;
            estims{li,nti}.hrs_sh=0;
            estims{li,nti}.hrs_ind=0;
            estims{li,nti}.xi=0;
        end
        %we need: q-model in which , up to q0(=0) is shuffled: model=q, shuffle=0
        %shuffle across:R (not S)
        
        if 1
            q=1;
            estims{li,nti}.hqrs=hqrs(spk2_dig,nts,q,methid);
            estims{li,nti}.hqr=hqr(spk2_dig,nts,q,methid);
            estims{li,nti}.xiq=xiq(spk2_dig,nts,q,methid);
            
            
            %spk_q=q_shuffle(spk,nts,q);
            spk_q=q_shuffle(spk2_dig,nts,q);
            %estims{li,nti}.hrs_q=hrs(spk_q,nts,methid);
            estims{li,nti}.hrs_q_sh=hrs(spk_q,nts,methid);
            
            %estims{li,nti}.hqrs0=hqrs(spk,nts,q,meth);
            %estims{li,nti}.hrss0=hrs_shuff(spk,nts,meth);
            %estims{li,nti}.spks=q_shuffle(spk,nts,q);
            %estims{li,nti}.hqrss0=hrs(spks,nts,meth);
            %estims{li,nti}.hr0=hr(spk,nts,meth);
            %estims{li,nti}.hqr0=hqr(spk,nts,q,meth);
            %estims{li,nti}.hrs0=hrs(spk,nts,meth);
            %estims{li,nti}.hrsi0=hrsind(spk,nmiu,meth);
            %estims{li,nti}.xiq0=xiq(spk,nmiu,q,meth);
            %estims{li,nti}.xi0=xi(spk,nmiu,meth);
        end
        
        estims{li,nti}.L=L;
        estims{li,nti}.len=lenL/L;
        estims{li,nti}.nt=nt;
        
        fprintf(']');
        
        fprintf(' (%g sec.)',floor(toc));
        fprintf('\n');
        %estims{li,nti}
    end
    
    %% -
    
    %a2d1_th=[];
    a2d1_d=[];
    a2d2_qd=[];
    a2d3_dsh=[];
    nta1d=[];
    for ntj=1:nti
        for li=1:length(LARR)
            %L=LARR(li);
            ee=estims{li,ntj};
            %mi=(ee.hr-ee.hrs)/ee.L;
            
            %if  L<=MAX_L_DIRECT
            if isfield(ee,'hrs_d')
                % mi1=(ee.hr-ee.hrs -ee.hrs_ind + ee.hrs_sh  )/ee.L;
                %mi1=(ee.hr_th-ee.hrs_th  )/ee.L;
                
                %mi1=(ee.hr_d-ee.hrs_d -ee.hrs_ind + ee.hrs_sh  )/ee.L;
                %mi2=(ee.hr_d-ee.hrs_d)/ee.L;
                mi1_plugin=(ee.hr_d-ee.hrs_d  )/ee.L;
                
                mi1_sh=(ee.hr_d-ee.hrs_d -ee.hrs_ind + ee.hrs_sh  )/ee.L;
                
            else
                mi1_plugin=0; %NaN;
                mi1_sh=0;
            end
            %a2d1_th(ntj,li)=mi1;
            a2d1_d(ntj,li)=mi1_plugin;
            a2d3_dsh(ntj,li)=mi1_sh;
            %mi2=(ee.xi-ee.hrs_ind)/ee.L;
            %mi2=(ee.hqr-ee.hrs -ee.hqrs + ee.hrs_q  )/ee.L;
            %mi2=(ee.xiq-ee.hrs_q +ee.hrs_sh - ee.hrs_ind  )/ee.L; %DID NOT
            %WORK
            %mi2=(ee.hqr-ee.hrs_q)/ee.L; %+ee.hrs_sh - ee.hrs_ind  )/ee.L;
            %slightly biased (uses hrs_q-sh)
            
            % ************ CHECK THIS *************
            mi2=(ee.xiq-ee.hqrs)/ee.L;
            %mi2=(ee.hqr-ee.hqrs)/ee.L;
            % ... %+ee.hrs_sh - ee.hrs_ind  )/ee.L;
            %mi2=(ee.xi-ee.hrs_ind)/ee.L;
            %mi2=(ee.hr1-ee.hrs1)/ee.L;
            %mi2=(ee.hr_d-ee.hrs_d)/ee.L;
            a2d2_qd(ntj,li)=mi2;
            %mi3=(ee.hqr-ee.hrs_q)/ee.L; %+ee.hrs_sh - ee.hrs_ind  )/ee.L;
            %mi3=(ee.xiq-ee.hrs_q)/ee.L;
            
            nta1d(ntj)=ee.nt;
            
        end
    end
    
    figure(5);clf;
    ax=[];
    %ax(1)=subplot(1,3,1:2);
    ax(1)=subplot(2,3,1);
    hold on; %box on;
    lha=[];
    
    %h = plot(nta1d,a2d1_th/binlen_sec);
    h = plot(nta1d,a2d1_d/binlen_sec);
    %if isempty(h)
    %    h=plot(0,0,'w.');
    %end
    set(h,'linewidth',2);
    
    set(h,'Displayname','L-direct'); %L-bin
    %set(h,'Displayname','dithered'); %L-bin
    
    %lha(1)=h(1);
    
    
    ylabel('Information rate [bits/sec]');
    % set(gca,'xscale','log');
    
    title('direct plug-in');
    
    %MARKOV
    ax(3)=subplot(2,3,3);
    
    h=plot(nta1d,a2d2_qd/binlen_sec,'-');
    
    %if isempty(h)
    %    h=plot(0,0,'w.');
    %end
    %plot(nta1d,a2d3,':');
    set(h,'Displayname','q-Markov'); %L-bin
    %set(h,'Displayname','L-direct');
    set(h,'linewidth',2);
    
    %lha(2)=h(1);
    
    %  q=NaN;
    %title('q=1');
    
    
    %plot((ee.nt),mi,'.');
    %text((ee.nt),mi, sprintf(' L=%d',ee.L));
    %   set(gca,'xscale','log');
    
    %end
    %  end
    %hold on;
    
    %plot(nta1d, ami_perbin + nta1d*0, 'k:');
    %nta1d+100
    %     nta1d_temp=myminmax(nta1d);  nta1d_temp(end)=nta1d(end)+100; nta1d_temp(1)=90;
    %     h=plot(nta1d_temp, ami_bps + nta1d_temp*0, 'k-.');
    %     set(h,'Displayname','analytical');
    %lha(3)=h;
    
    hold on;
    %    plot(nta1d, nta1d*0, 'k:');
    %    xlabel('Trials');
    %ylabel('Information [bits/bin]');
    
    %xlim([10,(max(NTA)+100)*1.1]);
    
    %xlim([min(NTA)-2,(max(NTA)+100)*1.1]);
    
    title('q-Markov');
    for li=1:length(LARR)
        %   text(nta1d(1),a2d1_d(1,li)/binlen_sec, sprintf(' L=%d',LARR(li)));
        lta{li}=sprintf(' L=%d',LARR(li));
    end
    lh=legend(lta);
    set(lh,'box','off');
    set(lh,'Location','SouthWest');
    set(lh,'FontSize',7);
    
    
    
    
    % PANEL 2
    
    %    h=legend(lha,'location','SouthEast'); set(h,'box','off');
    
    
    ax(2)=subplot(2,3,2);
    
    h=plot(nta1d,a2d3_dsh/binlen_sec,'-');
    %if isempty(h)
    %    h=plot(0,0,'w.');
    %end
    set(h,'Displayname','direct-sh'); %L-bin
    lha(3)=h(1);
    % set(gca,'xscale','log');
    set(h,'linewidth',2);
    
    title('direct-sh');
    
    
    
    
    
    for pani=1:3
        set(gcf,'CurrentAxes',ax(pani));
        hold on;
        nta1d_temp=myminmax(nta1d);  nta1d_temp(end)=nta1d(end)+100; nta1d_temp(1)=20; %90;
        %h=plot(ax(pani),
        h=plot(nta1d_temp, ami_bps + nta1d_temp*0, 'k--');
        set(h,'Displayname','analytical');
        set(gca,'xscale','log');
        plot(nta1d, nta1d*0, 'k:');
        xlabel('Trials');
        
        set(gca,'xtick',unique([2.^[5,7,9,log2(max(NTA))]]));
        ylim([-50,100]);
        box off;
    end
    linkaxes(ax, 'y');
    linkaxes(ax, 'x');
    nta1d_temp=myminmax(nta1d);
    nta1d_temp(end)=nta1d(end)+1000;
    nta1d_temp(1)=nta1d_temp(1)-4;
    xlim(nta1d_temp);
    a=[a2d3_dsh(:)/binlen_sec ; a2d1_d(:)/binlen_sec ;  a2d2_qd(:)/binlen_sec ];
    %ylim(myminmax(a)*1.5);
    
    
    
    
    
    %ax(4)=...
    subplot(2,3,4); hold on;
    
    WARR=LARR*binlen_sec*1000;
    %bar(LARR,a2d2_qd(end,:) /binlen_sec);
    plot(WARR,a2d2_qd(end,:) /binlen_sec,'ko-','displayname','q-Markov');
    
    plot(WARR,a2d1_d(end,:) /binlen_sec,'r--','displayname','direct-sh');
    LA2=[WARR,WARR(end)+1];
    h=plot(LA2, ami_bps + LA2*0, 'k--');
    set(h,'Displayname','analytical');
    %xlabel('L');
    xlabel('W [msec]');
    
    lh=legend('show');
    set(lh,'box','off');
    set(lh,'Location','NorthEast');
    %set(lh,'FontSize',7);
    
    %linkaxes(ax, 'y')
    
    %ylabel('Information rate [bits/sec]');
    
    xlim([0.1,max(LA2)+1]);
    ylabel('Information rate [bits/sec]');
    
    
    title_on_top2(sprintf('q=%d, DITHER, QE=%d, W=%g[ms]        \\tau_s,\\tau_n=%g,%g [ms]',q,methid,1000*binlen_sec,tau_s,tau_n),0);
    
    %darbareye hichi be ghatiyat nemirese. va in kheili Pernicious, va
    %PErvesaive, va daaem va hart dafe hastesh. bayad MM javab bekhad ta
    %begam.
    
    %save test_lor4???_all
    % my_save_png('test_lor4f1---dither-v2',{},[7,5]);
    
    
    pause(0.2);
end
finishtime=now;
toc
'.'


%% - plots
if 0
    DOUBLEPANEL=0;
    figure(6);clf;
    if DOUBLEPANEL
        subplot(1,2,1)
    end
    hold on;%box on;
    styc='rgbmkckkk';
    
    for li=1:length(LARR)
        lnt=[];mi=[];
        for ntj=1:nti
            ee=estims{li,ntj};
            
            %mi(ntj)=(ee.hr-ee.hrs)/ee.L;
            %mi(ntj)=(ee.hr-ee.hrs -ee.hrs_ind + ee.hrs_sh  )/ee.L;
            %mi(ntj)=(ee.hr1-ee.hrs1 -ee.hrs_ind + ee.hrs_sh  )/ee.L;
            mi(ntj)=(ee.hr_d-ee.hrs_d -ee.hrs_ind + ee.hrs_sh  )/ee.L;
            
            %lnt(ntj)=log2(ee.nt);
            lnt(ntj)=1./(ee.nt);
            plot(lnt(ntj),mi(ntj),[styc(li),'o']);
            text(lnt(ntj),mi(ntj), sprintf(' L=%d',ee.L));
        end
        h=plot(lnt,mi,[styc(li),'--']);
        set(h,'LineWidth',2);
        h1(li)=h;
    end
    
    %set(gca,'xlim',[4,1+log2(max(NTA))]);
    
    ami_perbin=ami_bps* binlen_sec;
    plot(lnt, ami_perbin + lnt*0, 'k--');
    
    xlabel('log_2(trials)');
    ylabel('information (bits)');
    ylim=get(gca,'ylim');
    %set(gca,'ylim',[0,ylim(2)]);
    %set(gca,'ylim',[0,0.5]);
    title('direct method');
    %  legend(h1, cell_sprintf('L=%d',LARR));
end
%%
%prs_dig(1,:)=[];
%%

figure(4);clf;
plot(resp_cont(1,:))
hold on;
plot(resp_dig(1,:)-0,'r')

figure(10);clf;
mypcolor(prs_dig(1:end,200:220)); title('trunc');
figure(11);clf;
mypcolor(prs_dith(:,200:220));title('dith');

figure(12);clf;
hist([prs_dith(:),prs_dig(:)])

%%
figure(13);clf;
a=resp_cont(1,:);
b=resp_dig(1,:)-0;
c=r2_raw(1,:);
plot(a,b+randn(size(b))*0.1,'k.');

mm=myminmax(c(:)+randn(size(a(:)))*0.01);
cx=mm(1):diff(mm)/40:mm(2);
h1=histc(c(:),cx);
h1=h1/sum(h1);
% *mean(diff(cx))
figure(14);clf;
subplot(1,2,1);
plot(c,a+randn(size(a))*0.05,'k.', 'markersize',1);
hold on;
plot(cx,h1*20+0.5,'r');
subplot(1,2,2);
plot(c,b+randn(size(a))*0.05,'k.', 'markersize',1);
hold on;
plot(cx,h1*20+0.5,'r');
for i=1:2
    subplot(1,2,i);
    
    for x0=[-4,4]
        plot(x0+[0,0],[0,M],'b--', 'linewidth',1);
    end
end

% my_save_png('test_lor4e2-v13-meth=1',{},[5,4]);

%%
%close all
figure(101);clf;
plot(pr_dig, 'displayname','discrete');
hold on;
plot(pr_dith,'r.-', 'displayname','dith');
[sum(pr_dith),sum(pr_dig)]
legend show;
"""