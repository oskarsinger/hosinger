% Test Cross-modality Event Discovery 
%A. Hero, July 2016
close all
clear all
plotting=0;wavelet_comp=1;
% Specify type of signal
%{
jitter_series=1; %if 0 the period is jittered by proportion prop_jitter
prop_jitter=0.0;
randomscaling=1;%if =0 then model is Xn=S+Nn
sigma_s1=0.5; %standard deviation of signal. Use when model is Xn=an S+Nn, an is i.i.d.
sigma_s2=0.5;
sigma_n=0.001;
%}
% Specify type of analysis
cov_analysis=1; %if cov_analysis = 0 then mean is not subtracted from sample. Use when model is Xn=S+Nn
% Specify operating parameters for time series (TS)
Ts=60*8;%sampling period (in minutes)
Tp=24;%Cyclic period in hours
N=8;%Number of periods in record
% Translate above params to index domain 
T=floor(Tp*60/Ts);%Cyclic period in index domain. 
Tpts=N*T; %Total number of time points
supp1=0.1;supp2=0.2; %True support (as proportional to T)of the correlated components in TS1 and TS2
%N=ceil(Tpts/T);
%Specify operation params for analysis
s1=1;s2=1; %Analysis window widths must be less than Tpts/T-cov_analysis-1
%Check rank condition
if cov_analysis==1
    display(['Covariance analysis: max(s1,s2)=',num2str(max(s1,s2)),'. Tpts/T-2=',num2str(Tpts/T-2)])
    if max(s1,s2)>Tpts/T-2
        display('Singular covariance! Reduce s1,s2 to less than Tpts/T-1');
    end
else
    display(['Mean square analysis: max(s1,s2)=',num2str(max(s1,s2)),'. Tpts/T-1=',num2str(Tpts/T-1)])
    if max(s1,s2)>Tpts/T-1
        display('Singular covariance! Reduce s1,s2 to less than Tpts/T');
    end
end
%{
% Generate two time series
cycl1=[];cycl2=[];
for k=1:N
rand_offset=round((rand(1,1)-0.5)*prop_jitter*T); %Generate jitter offset 
sig1=(1:floor(T*supp1));sig1=sig1/sqrt(sig1*sig1');
sig2=ones(1,floor(T*supp2));sig2=sig2/sqrt(sig2*sig2');
if randomscaling==1
    cycl1=[cycl1,sigma_s1*abs(randn(1,1))*sig1,zeros(1,ceil(T*(1-supp1))+rand_offset)];
    %cycl2=[(1:floor(T/supp2)),zeros(1,ceil(T*(1-1/supp2)))];
    cycl2=[cycl2,sigma_s2*abs(randn(1,1))*sig2,zeros(1,ceil(T*(1-supp2))+rand_offset)];
else
    cycl1=[cycl1,sig1,zeros(1,ceil(T*(1-supp1))+rand_offset)];
    %cycl2=[(1:floor(T/supp2)),zeros(1,ceil(T*(1-1/supp2)))];
    cycl2=[cycl2,sig2,zeros(1,ceil(T*(1-supp2))+rand_offset)];
end
end
%Now reshape to  N*T length time series
if length(cycl1)>N*T
    cycl1=cycl1(1:N*T);cycl2=cycl2(1:N*T);
else
    cycl1=[cycl1,zeros(1,N*T-length(cycl1))];cycl2=[cycl2,zeros(1,N*T-length(cycl2))];
end
%Generate noisy version
    TS1=cycl1+sigma_n*randn(1,N*T);
    TS2=cycl2+sigma_n*randn(1,N*T);
    
%}
addpath('~/Dropbox/Matlab/Scripts/')
[S, C, M] = csvimport('~/Data/CortisolMelatonin/cortisol_final.csv', 'columns', {'subject', 'cortisol', 'melatonin'});
TS1 = C(S == 2);
TS2 = M(S == 2);

%TS1=TS1(1:Tpts);TS2=TS2(1:Tpts);
figure
plot([TS1; TS2]'); title('Two modalities');xlabel('Sample')
legend(['TS1';'TS2'])

%Perform analysis
for t=1:T-max(s1,s2);
%Create windowed matrices X1 and X2 (cols are variables and rows are samples)
for i=1:N
    X1(i,:)=TS1(t+(T*(i-1)):t+(T*(i-1)+s1));
    X2(i,:)=TS2(t+(T*(i-1)):t+(T*(i-1)+s2));
end
if cov_analysis==1
%Covariance Analysis
Sigma12=X1'*(eye(N)-ones(N,1)*ones(1,N)/N)*X2/(N-1);
Sigma1=cov(X1);%X1'*X1;
Sigma2=cov(X2);%X2'*X2;
else
Sigma12=X1'*X2/(N-1);
Sigma1=X1'*X1/(N-1);
Sigma2=X2'*X2/(N-1);    
end
Sigma1sqrt=chol(Sigma1);
Sigma2sqrt=chol(Sigma2);
[U,S,V]=svd((Sigma1sqrt)^(-1)*Sigma12*(Sigma2sqrt')^(-1));
h11=-Sigma1sqrt'*U(:,1);
h21=-Sigma2sqrt'*V(:,1);
h12=-Sigma1sqrt'*U(:,2);
h22=-Sigma2sqrt'*V(:,2);
h11mat(:,t)=h11;
h21mat(:,t)=h21;
if plotting==1
figure;
subplot(1,2,1),plot([h11,h21])
title('First filter')
subplot(1,2,2), plot([h12,h22])
title('Second filter')
end %plotting
Sscore_vec(t)=S(1,1);
PSscore_vec(t)=sqrt(sum(h11.^2)+sum(h21.^2));
end %for loop on t
figure
subplot(1,2,1), plot(Sscore_vec),
title('Canonical correlation score'),xlabel('Period sample')
subplot(1,2,2), plot(PSscore_vec), text(T,0,[num2str(Tp),' hours'])
xlabel('Period sample')
title('Filter norm score')
figure
subplot(1,2,1),surfc(h11mat), title('Filter for TS1'), xlabel('Period'), ylabel('Support'), zlabel('h1')
subplot(1,2,2),surfc(h21mat), title('Filter for TS2'), xlabel('Period'), ylabel('Support'), zlabel('h2')

% c11=-U(:,1);
% c21=-V(:,1);
% c12=-U(:,2);
% c22=-V(:,2);
% figure;
% subplot(1,2,1),plot([c11,c21])
% title('First component')
% subplot(1,2,2), plot([c12,c22])
% title('Second component')

%%%%%%%%%%% Compute Daubechies wavelet decomposotion of the signals%%%%%%%%%%%
if wavelet_comp==1
wecg=TS1;
%load wecg;
Nf=3; % the order decomposition (max decimation factor is 2^Nf)
[C,L] = wavedec(wecg,Nf,'sym4');
wavcoefs = detcoef(C,L,'dcells');
a3 = appcoef(C,L,'sym4');
%%%%%% Render the wavelet decomposition as a series of approximately time aligned plots
nw=numel(wecg)+2*Nf*(2^Nf);
cfsmatrix = zeros(nw,4);
l1=length(wecg);l2=length(wavcoefs{1});l3=length(wavcoefs{2});l4=length(wavcoefs{3});
c1=nw;c2=length(cfsmatrix(1:2:end,1));c3=length(cfsmatrix(1:4:end,2));c4=length(cfsmatrix(1:8:end,3));
cfsmatrix(1:2:end,1) = [zeros(1,floor((c2-l2)/2)),wavcoefs{1},zeros(1,ceil((c2-l2)/2))];
cfsmatrix(1:4:end,2) = [zeros(1,floor((c3-l3)/2)),wavcoefs{2},zeros(1,ceil((c3-l3)/2))];
cfsmatrix(1:8:end,3) = [zeros(1,floor((c4-l4)/2)),wavcoefs{3},zeros(1,ceil((c4-l4)/2))];
cfsmatrix(1:8:end,4) = [zeros(1,floor((c4-l4)/2)),a3,zeros(1,ceil((c4-l4)/2))];
wecgp= [zeros(1,floor((c1-l1)/2)),wecg,zeros(1,ceil((c1-l1)/2))];
figure
subplot(5,1,1)
plot(wecgp); title('Original TS1 Signal'); %xlabel('Time index');
axis tight;
for kk = 2:4
    subplot(5,1,kk)
    stem(cfsmatrix(:,kk-1),'marker','none','ShowBaseLine','off');
    ylabel(['D' num2str(kk-1)]);
    axis tight;
end
subplot(5,1,5);
stem(cfsmatrix(:,end),'marker','none','ShowBaseLine','off');
ylabel('A3'); xlabel('Sample');
axis tight;


wecg=TS1;
%load wecg;
Nf=3; % the order decomposition (max decimation factor is 2^Nf)
[C,L] = wavedec(wecg,Nf,'sym4');
wavcoefs = detcoef(C,L,'dcells');
a3 = appcoef(C,L,'sym4');
%%%%%% Render the wavelet decomposition as a series of approximately time aligned plots
nw=numel(wecg)+2*Nf*(2^Nf);
cfsmatrix = zeros(nw,4);
l1=length(wecg);l2=length(wavcoefs{1});l3=length(wavcoefs{2});l4=length(wavcoefs{3});
c1=nw;c2=length(cfsmatrix(1:2:end,1));c3=length(cfsmatrix(1:4:end,2));c4=length(cfsmatrix(1:8:end,3));
cfsmatrix(1:2:end,1) = [zeros(1,floor((c2-l2)/2)),wavcoefs{1},zeros(1,ceil((c2-l2)/2))];
cfsmatrix(1:4:end,2) = [zeros(1,floor((c3-l3)/2)),wavcoefs{2},zeros(1,ceil((c3-l3)/2))];
cfsmatrix(1:8:end,3) = [zeros(1,floor((c4-l4)/2)),wavcoefs{3},zeros(1,ceil((c4-l4)/2))];
cfsmatrix(1:8:end,4) = [zeros(1,floor((c4-l4)/2)),a3,zeros(1,ceil((c4-l4)/2))];
wecgp= [zeros(1,floor((c1-l1)/2)),wecg,zeros(1,ceil((c1-l1)/2))];
figure
subplot(5,1,1)
plot(wecgp); title('Original TS2 Signal'); %xlabel('Time index');
axis tight;
for kk = 2:4
    subplot(5,1,kk)
    stem(cfsmatrix(:,kk-1),'marker','none','ShowBaseLine','off');
    ylabel(['D' num2str(kk-1)]);
    axis tight;
end
subplot(5,1,5);
stem(cfsmatrix(:,end),'marker','none','ShowBaseLine','off');
ylabel('A3'); xlabel('Sample');
axis tight;
end; %plotting wavelet decomposition

%%%%%%%%%% Complex Wavelet decomposition as in Kingsbury demo
path(path,'C:\Users\hero\Dropbox\Matlab\Kingsbury\dtcwt_toolbox4');

nlevels=6;
log2length=log2(length(wecg));
x=[wecg,zeros(1,2^ceil(log2length)-length(wecg))]';
% Specify DT CWT filters here.
biort = 'near_sym_b';
qshift = 'qshift_b';

% Forward DT CWT on each column of x.
[Yl,Yh,Yscale] = dtwavexfm(x,4,biort,qshift);
% Inverse DT CWT, one subband at a time, using gain_mask to select subbands.
z1 = dtwaveifm(Yl*0,Yh,biort,qshift,[1 0 0 0]);
z01 = dtwaveifm(Yl*0,Yh,biort,qshift,[0 1 0 0]);
z001 = dtwaveifm(Yl*0,Yh,biort,qshift,[0 0 1 0]);
z0001 = dtwaveifm(Yl*0,Yh,biort,qshift,[0 0 0 1]);
z0000 = dtwaveifm(Yl,Yh,biort,qshift,[0 0 0 0]);

% Check for perfect reconstruction: abs(error) < 1e-12.
z = z1 + z01 + z001 + z0001 + z0000;
DTCWT_error = max(abs(z(:)-x(:)))


%setfig(1)
% set(gcf,'DefaultTextFontSize',12,'Color',[1 1 1]);
% set(gcf,'numbertitle','off','name',['Step responses at 16 adjacent shifts']);
% subplot('position',[0.15 0.1 0.3 0.8]);
%sc = 1.2; % Scale offsets so sets of curves do not overlap.
%on = ones(size(x,2)-1,1)/5; 
%offset = sc*cumsum([0;on;1;on;1;on;1;on;1;on;1;on]/4).'; % Offsets for each curve.
zdtcwt = [x z1 z01 z001 z0001 z0000];% - ones(size(x,1),1)*offset;
[nr,nc]=size(zdtcwt);
%tt = 1:nr;  % Limits of plot horizontally.
tt=1:length(wecg); % truncate reconstructions to length of input
for i=1:nc
    subplot(nc,1,i), plot(tt,zdtcwt(tt,i),'-b');
    if i==1
        ylabel('Original')
    elseif i==nc
        ylabel('Scfn')
    else
        ylabel(['Level ',num2str(i-1)]);
    end
end
%axis off
%text(0,-7*sc,'(a) Dual Tree CWT','horiz','c')
% xpos = -42; % Position of text labels.
% text(xpos,0*sc,'Input','horiz','r','vert','m');
% text(xpos,-0.9*sc,'Wavelets','horiz','r','vert','m');
% text(xpos,-1.4*sc,'Level 1','horiz','r','vert','m');
% text(xpos,-2.4*sc,'Level 2','horiz','r','vert','m');
% text(xpos,-3.4*sc,'Level 3','horiz','r','vert','m');
% text(xpos,-4.4*sc,'Level 4','horiz','r','vert','m');
% text(xpos,-5.5*sc,'Scaling fn','horiz','r','vert','m');
% text(xpos,-6*sc,'Level 4','horiz','r','vert','m');
drawnow
