function p=probrind(spk,nt,range,f)
%this computes  P_ind(r)
%k indicates the fraction of trials to consider
%ntr=size(spk,3);
new_nt=floor(nt/f);
ns=size(spk,4);
L=size(spk,2);
M=max(reshape(spk,1,[]))+1;
if(M==1)
    M=2;
end
p=zeros(1,M^L);

if (L>1)
    
    twi=[0:M^L-1]'; #'
    swi=dec2base(twi,M);
    wi=zeros(M^L,L);
    for i=1:L
        wi(:,L-i+1)=str2num(swi(:,i));
    end


    %ntrk=size(range,2); 
    for t=1:ns
        
        spikes=(squeeze(spk(1,:,:,t)))'; %' trials set for current stimulus condition%now get the probabilities for each bin
        spikes=spikes(range(t,1:new_nt(t)),:);
        prob=zeros(L,M);
        for i=1:L
            n=spikes(:,i);
            Nmax=M-1;
            pbin=zeros(1,Nmax+1);
            count=zeros(Nmax+1,1);
            wi1=1+n;
            count=histc(wi1,[1:Nmax+1+eps]);
            
            pbin=(count/sum(count));
            lm=length(pbin);
            prob(i,1:lm)=pbin;    
        end
        pt=zeros(size(p));
        %size(wi),t,M
        for j=1:M^L
            
            pt(j)=prod(diag(prob([1:L],1+wi(j,:))));
                  
        end
        p=p+pt*new_nt(t);
    end

    p=p/sum(new_nt);
else
    p=probr(spk,nt,range,f);
end
