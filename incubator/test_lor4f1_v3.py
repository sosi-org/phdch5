
import numpy as np
import scipy
import math
import time

import matplotlib.pylab as plt

from makelorresponse import makelorresponse

from discr import discr
from dith_unsure import dith_unsure
from analytical_lor import analytical_lor

from hx_types import is_any_int_type
from q_shuffle import q_shuffle

MOCK=False

if not MOCK:
    from hr import hr
    from hrs import hrs
    #from my_hr import my_hr
    from range_shuffle import range_shuffle

    #print range_shuffle(np.array([5,3,1],dtype=int))
    #from hr import hr
    #from hr import hr
    #from hr import hr
    from hqrs import hqrs
    from hqr import hqr
    from xiq import xiq

    from hrs_shuff import hrs_shuff
    from hrsind import hrsind
    from xi import xi
    from    my_hrs import my_hrs
    from    my_prs_dith1 import my_prs_dith1
    from    my_pr import my_pr
else:
    import random
    def mock():
        return random.uniform(0.8,0.99)
    def hr(spk, nta, methid):
        return mock()
    def hrs(spk, nta, methid):
        return mock()
    def hqrs(spk2_dig, nts, q, methid):
        return mock()
    def hqr(spk2_dig, nts, q, methid):
        return mock()
    def xiq(spk2_dig, nts, q, methid):
        return mock()
    def hrs_shuff(spk2_dig, nts, methid):
        return mock()
    def hrsind(spk2_dig, nts, methid):
        return mock()
    def xi(spk2_dig, nts, methid):
        return mock()
    def my_hrs(prs_dith, ps_dith, methid):
        return mock()
    def my_pr(prs_dith, ps_dith):
        return mock()
    def my_hr(pr_dith, methid):
        return mock()
    def my_hrs(prs_dig, ps_dig, methid):
        return mock()
    def my_pr(prs_dig, ps_dig):
        return mock()
    def my_hr(pr_dig, methid):
        return mock()
    def my_prs_dith1(spk2_cont, nta):
        return mock(), mock()
    def my_prs(spk2_dig, nta):
        return mock(), mock()





def x():

    dithering = False
    #% from test_lor3c2
    #% 2) down sampling changes SNR
    tau_s = 25./1.
    std_s = 3.53
    tau_n = 12./1.
    std_n = 5.55
    methid = 0
    #%NTA=2.^[4:5];
    #%NTA=2.^[4:8,9];

    #NTA = np.power(2, np.array(np.hstack((np.arange(4, 7), 8, 10, 12))))
    #NTA = np.power(2, np.array([4, 6, 8])) #12
    NTA = np.power(2, np.array([4])) #12
    nt_generate = np.max(NTA)
    b_msec = 7.  #3,5,1
    fs_Hz = 1000.
    siglen_msec = 10000.
    #%kerlenratio=5;
    siglen_msec = math.floor(siglen_msec / b_msec)* b_msec
    #%uses fs_Hz, and then downsamples. The downsampling is not quite correct.
    downsample_b = int(b_msec/1000. * fs_Hz + 0.00001)
    #print downsample_b #7
    assert downsample_b == 7
    #%happens to be equal to b, if fs=1000
    if False:
        r2_raw = makelorresponse(tau_s, std_s, tau_n, std_n, nt_generate, siglen_msec, fs_Hz, [], downsample_b)
    else:
        #test only
        r2_raw=np.random.randn(nt_generate, int((siglen_msec / b_msec))) #temporary
    if False:
        print    r2_raw.shape #1428x256
        print nt_generate
        print nt_generate
        print nt_generate
    del nt_generate
    
    siglen_b = (siglen_msec / b_msec)
    del siglen_msec
    #%M=4;
    #%M=8;
    #%M=4;
    M = 5
    #%M=8;
    #%M=15;
    #%M=3;
    [resp_dig, dithinfo1] = discr(r2_raw, M)
    #%not dithered
    #assert is_any_int_type(r2_raw[0,0])
    #r2_raw ois float

    [resp_dith, dithinfo2] = dith_unsure(r2_raw, (dithinfo1['centres']))
    #%not dithered
    
    resp_dig = resp_dith
    #del dithinfo

    resp_dig = resp_dig-1
    assert is_any_int_type(resp_dig[0,0])
    resp_cont = r2_raw
    sz=resp_cont.shape
    #print sz #1428 * 256
    resp_cont=resp_cont.flatten(1)
    resp_cont = resp_cont-np.mean(resp_cont.flatten(1))
    #%resp_cont=resp_cont/std(resp_cont(:))/2;
    #%resp_cont=resp_cont/std(resp_cont(:))/2*1.5*4*4;
    #resp_cont = matdiv(np.dot(matdiv(resp_cont, np.std(resp_cont.flatten(1)))/2.*1., M), M-1.)
    #resp_cont=resp_cont/std(resp_cont(:))/2*1*M/(M-1);
    resp_cont =  resp_cont / np.std(resp_cont.flatten()) /2.* M / (M-1.)
    #% [-1,+1]
    resp_cont = (resp_cont+1.)/2.
    #% [0,1]
    resp_cont = resp_cont * (M-1.)
    if 1:
        resp_cont[(resp_cont<0.)] = 0
        resp_cont[(resp_cont > M-1.)] = M-1
    resp_cont = resp_cont.reshape(sz)
    



    #print "HERE"
    #exit(0)
    
    #%clear r2_raw;
    #% analytical formula
    #%fs_Hz is used for Nyquist frequency
    #%ami=analytical_lor(tau_s,std_s, tau_n, std_n, fs_Hz, 0.0001);
    #%ami_perbin=mi/fs_Hz*b_msec;
    #% %WHY?!
    #%a=analytical_lor(tau_s/b_msec,std_s, tau_n/b_msec, std_n, fs_Hz/b_msec, 0.0001);
    #%ami_perbin=a / (fs_Hz/b_msec);
    #% (tau_s_msec,  sigma_s,  tau_n_msec,  sigma_n, fs_Hz, d_freq)
    #%Ia_bits_per_sec
    #% Ia_bps --> ami_bps = analytical mi
    ami_bps = analytical_lor(tau_s, std_s, tau_n, std_n, (1./b_msec/1000.), 0.0001)
    #%ami_perbin=ami_bps* (b_msec/1000);
    #% 1/ (fs_Hz/b_msec) = b_msec / fs = above (bcoz, fs=1000)
    #% previous: 0.2362
    #% new: 2.4512e-04
    binlen_sec = b_msec/1000.
    #%used later for normalization
    
    #del a
    del b_msec

    #%clearvars -except   resp_cont resp_dig M NTA binlen_sec siglen_b tau_s tau_n ami_bps ...
    #%    r2_raw std_* methid;
    don_plot_this = False
    #%size(respcut_dig)
    if don_plot_this:
            #%%
            #%mean(mean(resp_dig,1))
            [c, d] = plt.xcorr((matdiv(np.mean(resp_dig, 1.)-1.-(M-1.)/2.+1., M)*2.), 'unbiased')
            plt.figure(10.)
            plt.clf
            #% / sqrt(std_s^2+std_n^2)
            h = plt.plot((np.dot(d, binlen_sec)*1000.), matdiv(c, std_s**2.), 'k.-')
            set(h, 'linewidth', 2.)
            set(h, 'displayname', 'signal')
            plt.xlim((np.array(np.hstack((-1., 1.)))*100.*1.))
            #%set(gca,'xtick',-200:10:200);
            plt.title('\\tau_s, \\tau_n = %g,%g [msec]'%( tau_s, tau_n))
            #nn = matcompat.size(resp_dig, 1.)
            nn = resp_dig.shape[1-1]
            c1 = 0.
            for tri in np.arange(1., (nn)+1):
                #%mean(resp_dig(tri,:))
                #MISSING CODE!!!
                [c,d]=np.xcorr((resp_dig[tri,:]-1-(M-1)/2+1)/M*2,'unbiased');
                c1=c1+c/nn;
                pass

            del nn
            plt.hold(on)
            h = plt.plot((np.dot(d, binlen_sec)*1000.), matdiv(c1, std_n**2.), 'r.-')
            set(h, 'linewidth', 2.)
            set(h, 'displayname', 'noise')
            tt = np.dot(d, binlen_sec)
            h = plt.plot((tt*1000.), np.dot(binlen_sec, np.exp(matdiv(-np.abs(tt), tau_s/1000.))), 'k--')
            set(h, 'displayname', 'signal analytical')
            h = plt.plot((tt*1000.), np.dot(binlen_sec, np.exp(matdiv(-np.abs(tt), tau_n/1000.))), 'r--')
            set(h, 'displayname', 'noise analytical')
            h = plt.legend('show')
            set(h, 'box', 'off')
            set(h, 'location', 'NorthWest')
            plt.xlabel('Time [msec]')
            plt.title('cross correlation')
            #% my_save_png('test_lor4f1---signal-xcorr-v12-temp',{},[4,3]);
            
    begintime = time.time()
    #tic=time.time()

    #%estims=cell(4,4);
    #%LARR=[1,4,8,10];
    #%LARR=[1,2:2:10];
    #%LARR=[1,2:2:8];
    #%LARR=[1,2,3,6];
    #%LARR=[1,2,3,6,7];
    #%LARR=[1]; %
    #LARR = np.array(np.hstack((1., 3., 6., 7.)))
    #LARR = np.array(np.hstack((1+1, 3, 6, 7)))
    #LARR = [1+1, 3, 6, 7]
    LARR = [1,1+1, 4]
    MAX_L_DIRECT = np.Inf

    #estims_cell=[[],[],[],[]]
    #estims_cell=[[{}],[{}],[{}],[{}]]
    #estims_cell=[[{}]]
    estims_cell=[]
    #len(NTA)
    #print len(LARR),len(NTA), "******************"
    for i in range(0,len(LARR)):
            estims_cell.append([])
            for j in range(0,len(NTA)):
                print i,j
                estims_cell[i].append({})


    #%for nti=10:len(NTA)
    for nti in range(1, len(NTA)+1):
        nt = int(NTA[int(nti)-1])
        for li in range(1, len(LARR)+1):
            L = LARR[int(li)-1]
            #%lenL=floor(siglen_msec/b_msec/L)*L;
            #%lenL=floor(siglen_msec/L/b_msec)*L*b_msec;
            lenL = (int((siglen_b/ L)) * L)
            #%respcut=resp(:,1:lenL)';
            #%  respcut=resp(1:nt,1:lenL)';
            if dithering:
                respcut_cont = resp_cont[0:nt,0:lenL].conj().T


            #print "HERE"
            #exit(0)
            
            #print nt, max(NTA) #16,256
            #print resp_cont.shape  #1428x256
            #print resp_dig.shape
            #print "000"
            respcut_dig = resp_dig[0:nt,0:lenL].conj().T
            #% len x trials
            #%spk0 = zeros(1,LNum,size(signal,1), newlen );
            #%spk=reshape(respcut,  [1,L,size(resp)]);
            #% spk__=reshape(respcut,  1,L,lenL/L,[]);
            #%dithering = False
            if dithering:
                print "Using dithering"
                #spk__cont = np.reshape(respcut_cont, 1., L, matdiv(lenL, L), np.array([]))
                spk__cont = np.reshape(respcut_cont, [1, L, int(lenL/ L), -1])
                #% 1 x L x len x trials
                #spk2_cont = permute(spk__cont, np.array(np.hstack((1., 2., 4., 3.))))
                spk2_cont = spk__cont.transpose([1-1, 2-1, 4-1, 3-1]) #, np.array(np.hstack((1., 2., 4., 3.))))
                assert spk2_cont.shape == (1, L, nt, int(lenL/ L))

                #clear(spk__c)
                # #%slow
            else:
                print "Not using dithering"
            
            
            #% 1 x L x trials x len
            #print respcut_dig.shape, "*1"  #256 x 16  
            #print (lenL,L,nt), "*2" #1428.0, 1.0, 16
            spk__dig = np.reshape(respcut_dig, [1, L, int(lenL/ L), -1])
            spk2_dig = spk__dig.transpose([1-1, 2-1, 4-1, 3-1]) #permute(spk__dig, np.array(np.hstack((1., 2., 4., 3.))))
            assert spk2_dig.shape == (1, L, nt, int(lenL/ L))
            assert is_any_int_type(spk2_dig[0,0,0,0])

            del spk__dig

            #%slow
            
            assert type(nt) is int
            #nts = np.dot(np.ones(1., matdiv(lenL, L)), nt)
            #nts = np.ones([(lenL/ L)]) * nt
            nts = np.array([nt]*(lenL/ L), dtype=int) #np.ones([(lenL/ L)]) * nt
            #[nt]*int(lenL/ L)

            #print nts

            #assert type(nts[0]) is int
            #print type(nts), type(nts[0])
             

            
            print 'spk2: %dx%dx%dx%d ['%spk2_dig.shape
            if L<=MAX_L_DIRECT:
                ns = int(lenL/ L)
                #nta = np.arange(1., (ns)+1)*0.+nt
                nta = np.array([nt]*ns)
                if dithering:
                    #%[prs3,ps3]=my_prs_dith1(spk2, nta);
                    #%   [prs3_dith,ps3]=my_prs_dith1(spk2_c-1, nta);
                    #%[prs3_dith,ps3]=my_prs_dith1(spk2_cont-0, nta);
                    (prs_dith, ps_dith) = my_prs_dith1((spk2_cont-0.), nta)
                    #%    [prs3_dig,ps3]=my_prs(spk1, nta);
                    #%    sum(abs(prs3_dith(:)-prs3_dig(:)))
                    #%   if 0
                    #%       %[prs1,ps1]=my_prs(spk2, nta);
                    #%       [prs1,ps1]=my_prs(spk2_dig, nta);
                    #%   end
                    (prs_dig, ps_dig) = my_prs(spk2_dig, nta)
                    #print(repr((prs_dig, ps_dig)))
                    #%prs=prs3_dith;
                    #%ps=ps3;
                    fprintf('p')
                    hrs0_th = my_hrs(prs_dith, ps_dith, methid)
                    pr_dith = my_pr(prs_dith, ps_dith)
                    hr0_th = my_hr(pr_dith, methid)
                    #%hr1=hr(spk2_dig,nts,methid);
                    #%hrs1=hrs(spk2_dig,nts,methid);
                    hrs_dig = my_hrs(prs_dig, ps_dig, methid)
                    pr_dig = my_pr(prs_dig, ps_dig)
                    hr_dig = my_hr(pr_dig, methid)
                    #%[ my_hr(ps_dith, methid),my_hr(ps_dig, methid)]
                    #%m22= [...
                    #%    my_hr(pr_dith, methid),my_hr(pr_dig, methid);
                    #%    my_hrs(prs_dith,ps_dith, methid),my_hrs(prs_dig,ps_dig, methid)];
                    #%
                    vv1 = my_hr(pr_dith, methid)
                    my_hr(pr_dig, methid)
                    vv2 = my_hrs(prs_dith, ps_dith, methid)
                    my_hrs(prs_dig, ps_dig, methid)
                    m22 = vertcat(vv1, vv2)
                    #m22

                    #print li, nti
                    #%myminmax(spk2_dig(:))  % 1--4
                    #%myminmax(spk2_cont(:)) % 0--3
                    #estims.cell[int(li)-1,int(nti)-1]['hr_d'] = hr_dig
                    estims_cell[int(li)-1][int(nti)-1]['hr_d'] = hr_dig
                    #%hr1;
                    estims_cell[int(li)-1][int(nti)-1]['hrs_d'] = hrs_dig
                    #%hrs1;
                    estims_cell[int(li)-1][int(nti)-1]['hr_th'] = hr0_th
                    estims_cell[int(li)-1][int(nti)-1]['hrs_th'] = hrs0_th


                #print type(nts), type(nts[0])
                hr1_dig = hr(spk2_dig, nts, methid)
                hrs1_dig = hrs(spk2_dig, nts, methid)
                #print li, nti #1.0, 1.0
                #print int(li)-1,int(nti)-1
                #print estims_cell[int(li)-1][int(nti)-1]
                zz=estims_cell[int(li)-1][int(nti)-1]
                zz['hr_d'] = 1
                #print hr1_dig
                estims_cell[int(li)-1][int(nti)-1]['hr_d'] = hr1_dig
                #%hr1;
                estims_cell[int(li)-1][int(nti)-1]['hrs_d'] = hrs1_dig
                #%hrs1;
            
    
                if 1:

                    #print "***********"
                    #print nts
                    #%no dithering
                    estims_cell[int(li)-1][int(nti)-1]['hrs_sh'] = hrs_shuff(spk2_dig, nts, methid)

                    estims_cell[int(li)-1][int(nti)-1]['hrs_ind'] = hrsind(spk2_dig, nts, methid)
                    estims_cell[int(li)-1][int(nti)-1]['_xi'] = xi(spk2_dig, nts, methid)

                    #    
    
            else:
                estims_cell[int(li)-1][int(nti)-1]['_hr'] = 0.
                estims_cell[int(li)-1][int(nti)-1]['_hrs'] = 0.
                estims_cell[int(li)-1][int(nti)-1]['hrs_sh'] = 0.
                estims_cell[int(li)-1][int(nti)-1]['hrs_ind'] = 0.
                estims_cell[int(li)-1][int(nti)-1]['_xi'] = 0.
                
            
            #%we need: q-model in which , up to q0(=0) is shuffled: model=q, shuffle=0
            #%shuffle across:R (not S)
            #%if 1
            #print "HERE2"
            #exit(0)


            q = 1
            estims_cell[int(li)-1][int(nti)-1]['_hqrs'] = hqrs(spk2_dig, nts, q, methid)
            estims_cell[int(li)-1][int(nti)-1]['_hqr'] = hqr(spk2_dig, nts, q, methid)
            estims_cell[int(li)-1][int(nti)-1]['_xiq'] = xiq(spk2_dig, nts, q, methid)
            #%spk_q=q_shuffle(spk,nts,q);

            spk_q = q_shuffle(spk2_dig, nts, q)
            #%estims{li,nti}['hrs_q']=hrs(spk_q,nts,methid);
            estims_cell[int(li)-1][int(nti)-1]['hrs_q_sh'] = hrs(spk_q, nts, methid)
            #%estims{li,nti}.hqrs0=hqrs(spk,nts,q,meth);
            #%estims{li,nti}.hrss0=hrs_shuff(spk,nts,meth);
            #%estims{li,nti}.spks=q_shuffle(spk,nts,q);
            #%estims{li,nti}.hqrss0=hrs(spks,nts,meth);
            #%estims{li,nti}.hr0=hr(spk,nts,meth);
            #%estims{li,nti}.hqr0=hqr(spk,nts,q,meth);
            #%estims{li,nti}.hrs0=hrs(spk,nts,meth);
            #%estims{li,nti}.hrsi0=hrsind(spk,nmiu,meth);
            #%estims{li,nti}.xiq0=xiq(spk,nmiu,q,meth);
            #%estims{li,nti}.xi0=xi(spk,nmiu,meth);
            #%end
            estims_cell[int(li)-1][int(nti)-1]['L'] = L
            estims_cell[int(li)-1][int(nti)-1]['len'] = lenL/ L
            estims_cell[int(li)-1][int(nti)-1]['nt'] = nt
            print(']'),
            #print' (%g sec.)'%(np.floor(toc),)
            print('\n')
            #%estims{li,nti}

        #Takes long up to here. Cleaned up.
        #print "HERE"
        #exit(0)

        #%% -
        #%a2d1_th=[];
        #a2d1_d = [] #np.array([])
        a2d1_d = [] #np.array([])
        a2d2_qd = [] #np.array([])
        a2d3_dsh = [] #np.array([])
        nta1d = [] #np.array([])
        for ntj in range(1, nti+1):
            a2d1_d.append([])
            assert len(a2d1_d)-1 == ntj-1 #assert index number
            a2d3_dsh.append([])
            assert len(a2d3_dsh)-1 == ntj-1 #assert index number
            a2d2_qd.append([])
            assert len(a2d2_qd)-1 == ntj-1 #assert index number

            #ee=estims_cell[li-1][ntj-1]
            #nta1d.append( ee['nt'] )

            nta1d.append( estims_cell[0][ntj-1]['nt'] )
            assert len(nta1d)-1 == ntj-1

            for li in range(1, len(LARR)+1 ):
                #%L=LARR(li);
                #ee=estims{li,ntj};
                print estims_cell  #list of list of dict
                #print estims_cell.shape
                print estims_cell[0][0]
                print li,ntj
                ee=estims_cell[li-1][ntj-1]
                #%if  L<=MAX_L_DIRECT
                if 'hrs_d' in ee: #haskey(ee, 'hrs_d'):
                    #% mi1=(ee['_hr']-ee['_hrs'] -ee['hrs_ind'] + ee['hrs_sh']  )/ee['L'];
                    #%mi1=(ee['hr_th']-ee['hrs_th']  )/ee['L'];
                    #%mi1=(ee['hr_d']-ee['hrs_d'] -ee['hrs_ind'] + ee['hrs_sh']  )/ee['L'];
                    #%mi2=(ee['hr_d']-ee['hrs_d'])/ee['L'];
                    assert not type(ee['hr_d']) is list
                    #mi1_plugin = matdiv(ee['hr_d']-ee['hrs_d'], ee['L'])
                    #mi1_sh = matdiv(ee['hr_d']-ee['hrs_d']-ee['hrs_ind']+ee['hrs_sh'], ee['L'])
                    mi1_plugin = (ee['hr_d']-ee['hrs_d']) / ee['L']
                    mi1_sh = (ee['hr_d']-ee['hrs_d']-ee['hrs_ind']+ee['hrs_sh']) /  ee['L']
                else:
                    mi1_plugin = 0.
                    #%NaN;
                    mi1_sh = 0.
                    
                
                print mi1_plugin
                #%a2d1_th(ntj,li)=mi1;
                #a2d1_d[int(ntj)-1,int(li)-1] = mi1_plugin
                a2d1_d[ntj-1].append(mi1_plugin)
                assert len(a2d1_d[ntj-1])==li-1+1
                #a2d3_dsh[int(ntj)-1,int(li)-1] = mi1_sh
                a2d3_dsh[ntj-1].append( mi1_sh )
                assert len(a2d3_dsh[ntj-1])==li-1+1
                
                #%mi2=(ee['_xi']-ee['hrs_ind'])/ee['L'];
                #%mi2=(ee['_hqr']-ee['_hrs'] -ee['_hqrs'] + ee['hrs_q']  )/ee['L'];
                #%mi2=(ee['_xiq']-ee['hrs_q'] +ee['hrs_sh'] - ee['hrs_ind']  )/ee['L']; %DID NOT
                #%WORK
                #%mi2=(ee['_hqr']-ee['hrs_q'])/ee['L']; %+ee['hrs_sh'] - ee['hrs_ind']  )/ee['L'];
                #%slightly biased (uses hrs_q-sh)
                #% ************ CHECK THIS *************
                #mi2 = matdiv(ee['_xiq']-ee['_hqrs'], ee['L'])
                mi2 = (ee['_xiq']-ee['_hqrs']) /  ee['L']
                #%mi2=(ee['_hqr']-ee['_hqrs'])/ee['L'];
                #% ... %+ee['hrs_sh'] - ee['hrs_ind']  )/ee['L'];
                #%mi2=(ee['_xi']-ee['hrs_ind'])/ee['L'];
                #%mi2=(ee.hr1-ee.hrs1)/ee['L'];
                #%mi2=(ee['hr_d']-ee['hrs_d'])/ee['L'];
                #a2d2_qd[int(ntj)-1,int(li)-1] = mi2
                a2d2_qd[ntj-1].append( mi2 )
                assert len(a2d2_qd[ntj-1]) -1 == li-1
                #%mi3=(ee['_hqr']-ee['hrs_q'])/ee['L']; %+ee['hrs_sh'] - ee['hrs_ind']  )/ee['L'];
                #%mi3=(ee['_xiq']-ee['hrs_q'])/ee['L'];
                #nta1d[int(ntj)-1] = ee['nt']
                #nta1d.append( ee['nt'] )
                #print len(nta1d),ntj
                #print nta1d
                #assert len(nta1d)-1 == ntj-1
                #assert len(nta1d)-1 == ntj-1  ##COMPILE_TIME:CHECK_PATTERN: MATCH LAST_INDEX(nta1d) == ntj-1   #PLANGNOTE
                assert nta1d[ntj-1] == ee['nt']
                #%    end
                #%end
               
        if False:
            plt.figure(5.)
            plt.clf
            ax = np.array([])
            #%ax(1)=subplot(1,3,1:2);
            ax[0] = plt.subplot(2., 3., 1.)
            plt.hold(on)
            #%box on;
            lha = np.array([])
            #%h = plot(nta1d,a2d1_th/binlen_sec);
            h = plt.plot(nta1d, matdiv(a2d1_d, binlen_sec))
            #%if isempty(h)
            #%    h=plot(0,0,'w.');
            #%end
            set(h, 'linewidth', 2.)
            set(h, 'Displayname', 'L-direct')
            #%L-bin
            #%set(h,'Displayname','dithered'); %L-bin
            #%lha(1)=h(1);
            plt.ylabel('Information rate [bits/sec]')
            #% set(gca,'xscale','log');
            plt.title('direct plug-in')
            #%MARKOV
            ax[2] = plt.subplot(2., 3., 3.)
            h = plt.plot(nta1d, matdiv(a2d2_qd, binlen_sec), '-')
            #%if isempty(h)
            #%    h=plot(0,0,'w.');
            #%end
            #%plot(nta1d,a2d3,':');
            set(h, 'Displayname', 'q-Markov')
            #%L-bin
            #%set(h,'Displayname','L-direct');
            set(h, 'linewidth', 2.)
            #%lha(2)=h(1);
            #%  q=NaN;
            #%title('q=1');
            #%plot((ee['nt']),mi,'.');
            #%text((ee['nt']),mi, sprintf(' L=%d',ee['L']));
            #%   set(gca,'xscale','log');
            #%end
            #%  end
            #%hold on;
            #%plot(nta1d, ami_perbin + nta1d*0, 'k:');
            #%nta1d+100
            #%     nta1d_temp=myminmax(nta1d);  nta1d_temp(end)=nta1d(end)+100; nta1d_temp(1)=90;
            #%     h=plot(nta1d_temp, ami_bps + nta1d_temp*0, 'k-.');
            #%     set(h,'Displayname','analytical');
            #%lha(3)=h;
            plt.hold(on)
            #%    plot(nta1d, nta1d*0, 'k:');
            #%    xlabel('Trials');
            #%ylabel('Information [bits/bin]');
            #%xlim([10,(max(NTA)+100)*1.1]);
            #%xlim([min(NTA)-2,(max(NTA)+100)*1.1]);
            plt.title('q-Markov')
            lta=[]
            for li in np.arange(1., (len(LARR))+1):
                #%   text(nta1d(1),a2d1_d(1,li)/binlen_sec, sprintf(' L=%d',LARR(li)));
                lta[li]=' L=%d'%(LARR(li),);
                pass
 
            if False:
                lh = plt.legend(lta)
                set(lh, 'box', 'off')
                set(lh, 'Location', 'SouthWest')
                set(lh, 'FontSize', 7.)
            #% PANEL 2
            #%    h=legend(lha,'location','SouthEast'); set(h,'box','off');
            ax[1] = plt.subplot(2., 3., 2.)
            h = plt.plot(nta1d, matdiv(a2d3_dsh, binlen_sec), '-')
            #%if isempty(h)
            #%    h=plot(0,0,'w.');
            #%end
            set(h, 'Displayname', 'direct-sh')
            #%L-bin
            lha[2] = h[0]
            #% set(gca,'xscale','log');
            set(h, 'linewidth', 2.)
            plt.title('direct-sh')
            for pani in np.arange(1., 4.0):
                set(plt.gcf, 'CurrentAxes', ax[int(pani)-1])
                plt.hold(on)
                nta1d_temp = myminmax(nta1d)
                nta1d_temp[int(0)-1] = nta1d[int(0)-1]+100.
                nta1d_temp[0] = 20.
                #%90;
                #%h=plot(ax(pani),
                h = plt.plot(nta1d_temp, (ami_bps+nta1d_temp*0.), 'k--')
                set(h, 'Displayname', 'analytical')
                set(plt.gca, 'xscale', 'log')
                plt.plot(nta1d, (nta1d*0.), 'k:')
                plt.xlabel('Trials')
                set(plt.gca, 'xtick', np.unique(np.array(np.hstack(((2 ** np.array(np.hstack((5., 7., 9., np.log2(matcompat.max(NTA)))))))))))
                plt.ylim(np.array(np.hstack((-50., 100.))))
                plt.box(off)
                
            linkaxes(ax, 'y')
            linkaxes(ax, 'x')
            nta1d_temp = myminmax(nta1d)
            nta1d_temp[int(0)-1] = nta1d[int(0)-1]+1000.
            nta1d_temp[0] = nta1d_temp[0]-4.
            plt.xlim(nta1d_temp)
            #%a=[a2d3_dsh(:)/binlen_sec ; a2d1_d(:)/binlen_sec ;  a2d2_qd(:)/binlen_sec ];
            a = vertcat(matdiv(a2d3_dsh.flatten(1), binlen_sec), matdiv(a2d1_d.flatten(1), binlen_sec), matdiv(a2d2_qd.flatten(1), binlen_sec))
            #%ylim(myminmax(a)*1.5);
            #%ax(4)=
            plt.subplot(2., 3., 4.)
            plt.hold(on)
            WARR = np.dot(LARR, binlen_sec)*1000.
            #%bar(LARR,a2d2_qd(end,:) /binlen_sec);
            plt.plot(WARR, matdiv(a2d2_qd[int(0)-1,:], binlen_sec), 'ko-', 'displayname', 'q-Markov')
            plt.plot(WARR, matdiv(a2d1_d[int(0)-1,:], binlen_sec), 'r--', 'displayname', 'direct-sh')
            LA2 = np.array(np.hstack((WARR, WARR[int(0)-1]+1.)))
            h = plt.plot(LA2, (ami_bps+LA2*0.), 'k--')
            set(h, 'Displayname', 'analytical')
            #%xlabel('L');
            plt.xlabel('W [msec]')
            lh = plt.legend('show')
            set(lh, 'box', 'off')
            set(lh, 'Location', 'NorthEast')
            #%set(lh,'FontSize',7);
            #%linkaxes(ax, 'y')
            #%ylabel('Information rate [bits/sec]');
            plt.xlim(np.array(np.hstack((0.1, matcompat.max(LA2)+1.))))
            plt.ylabel('Information rate [bits/sec]')
            title_on_top2(sprintf('q=%d, DITHER, QE=%d, W=%g[ms]        \\tau_s,\\tau_n=%g,%g [ms]', q, methid, (1000.*binlen_sec), tau_s, tau_n), 0.)
            #%darbareye hichi be ghatiyat nemirese. va in kheili Pernicious, va
            #%PErvesaive, va daaem va hart dafe hastesh. bayad MM javab bekhad ta
            #%begam.
            #%save test_lor4???_all
            #% my_save_png('test_lor4f1---dither-v2',{},[7,5]);
        time.sleep(0.2)
        print("sleep")
        
    finishtime = time.time() #now()
    #toc=time.time()
    print('.')


    print estims_cell
    maxc = 0
    collect_keys={}
    for i in range(0,len(estims_cell)):
        for j in range(0,len(estims_cell[i])):
            for k in estims_cell[i][j]:
               collect_keys[k]=0
               if maxc < j+1:
                    maxc = j+1
    print collect_keys.keys()
    print
    ka={}
    for k in collect_keys:
       ka[k]=np.zeros((len(estims_cell),maxc))

    for i in range(0,len(estims_cell)):
        for j in range(0,len(estims_cell[i])):
            for k in collect_keys:
               ka[k][i,j]=estims_cell[i][j][k]
    for k in collect_keys:
        print k+":",
        print ka[k].T
    return

    dont_plot_this = False
    #%% - plots
    if dont_plot_this:
        DOUBLEPANEL = 0.
        plt.figure(6.)
        plt.clf
        if DOUBLEPANEL:
            plt.subplot(1., 2., 1.)
        
        
        #plt.hold(on)
        #%box on;
        styc = 'rgbmkckkk'
        for li in np.arange(1., (len(LARR))+1):
            lnt = np.array([])
            mi = np.array([])
            for ntj in np.arange(1., (nti)+1):
                ee = estims_cell[int(li)-1,int(ntj)-1]
                #%mi(ntj)=(ee['_hr']-ee['_hrs'])/ee['L'];
                #%mi(ntj)=(ee['_hr']-ee['_hrs'] -ee['hrs_ind'] + ee['hrs_sh']  )/ee['L'];
                #%mi(ntj)=(ee.hr1-ee.hrs1 -ee['hrs_ind'] + ee['hrs_sh']  )/ee['L'];
                mi[int(ntj)-1] = matdiv(ee['hr_d']-ee['hrs_d']-ee['hrs_ind']+ee['hrs_sh'], ee['L'])
                #%lnt(ntj)=log2(ee['nt']);
                lnt[int(ntj)-1] = matdiv(1., ee['nt'])
                plt.plot(lnt[int(ntj)-1], mi[int(ntj)-1], np.array(np.hstack((styc[int(li)-1], 'o'))))
                plt.text(lnt[int(ntj)-1], mi[int(ntj)-1], sprintf(' L=%d', (ee['L'])))
                
            h = plt.plot(lnt, mi, np.array(np.hstack((styc[int(li)-1], '--'))))
            set(h, 'LineWidth', 2.)
            h1[int(li)-1] = h
            
        #%set(gca,'xlim',[4,1+log2(max(NTA))]);
        ami_perbin = np.dot(ami_bps, binlen_sec)
        plt.plot(lnt, (ami_perbin+lnt*0.), 'k--')
        plt.xlabel('log_2(trials)')
        plt.ylabel('information (bits)')
        ylim = plt.get(plt.gca, 'ylim')
        #%set(gca,'ylim',[0,ylim(2)]);
        #%set(gca,'ylim',[0,0.5]);
        plt.title('direct method')
        #%  legend(h1, cell_sprintf('L=%d',LARR));
    
    
    #%%
    #%prs_dig(1,:)=[];
    #%%
    plt.figure(4.)
    plt.clf
    plt.plot(resp_cont[0,:])
    #plt.hold(on)
    plt.plot((resp_dig[0,:]-0.), 'r')
    plt.figure(10.)
    plt.clf
    mypcolor(prs_dig[0:,199:220.])
    plt.title('trunc')
    plt.figure(11.)
    plt.clf
    mypcolor(prs_dith[:,199:220.])
    plt.title('dith')
    plt.figure(12.)
    plt.clf
    plt.hist(np.array(np.hstack((prs_dith.flatten(1), prs_dig.flatten(1)))))
    #%%
    plt.figure(13.)
    plt.clf
    a = resp_cont[0,:]
    b = resp_dig[0,:]-0.
    c = r2_raw[0,:]
    plt.plot(a, (b+np.dot(plt.randn(matcompat.size(b)), 0.1)), 'k.')
    mm = myminmax((c.flatten(1)+np.dot(plt.randn(matcompat.size(a.flatten(1))), 0.01)))
    cx = np.arange(mm[0], (mm[1])+(np.diff(mm)/40.), np.diff(mm)/40.)
    h1 = histc(c.flatten(1), cx)
    h1 = matdiv(h1, np.sum(h1))
    #% *mean(diff(cx))
    plt.figure(14.)
    plt.clf
    plt.subplot(1., 2., 1.)
    plt.plot(c, (a+np.dot(plt.randn(matcompat.size(a)), 0.05)), 'k.', 'markersize', 1.)
    plt.hold(on)
    plt.plot(cx, (h1*20.+0.5), 'r')
    plt.subplot(1., 2., 2.)
    plt.plot(c, (b+np.dot(plt.randn(matcompat.size(a)), 0.05)), 'k.', 'markersize', 1.)
    plt.hold(on)
    plt.plot(cx, (h1*20.+0.5), 'r')
    for i in np.arange(1., 3.0):
        plt.subplot(1., 2., i)
        for x0 in np.array(np.hstack((-4., 4.))):
            plt.plot((x0+np.array(np.hstack((0., 0.)))), np.array(np.hstack((0., M))), 'b--', 'linewidth', 1.)
            
        
    #% my_save_png('test_lor4e2-v13-meth=1',{},[5,4]);
    #%%
    #%close all
    plt.figure(101.)
    plt.clf
    plt.plot(pr_dig, 'displayname', 'discrete')
    plt.hold(on)
    plt.plot(pr_dith, 'r.-', 'displayname', 'dith')
    np.array(np.hstack((np.sum(pr_dith), np.sum(pr_dig))))
    plt.legend(show)

    return estims_cell






x()
