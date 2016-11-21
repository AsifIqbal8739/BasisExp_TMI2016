%%  Component extraction from the Signal Using Double Sparsity Algorithms
% Blind Source Separation code from the paper
clear all; close all; clc; 
tic
%% Data Stuff and Parameter Setup
m = 120;     n = 2;      g = 40; % N = size(Ya,1);
K = 1;       noIt = 20;
Trials = 100;
D_base = odctdict(m,g);    Aini = randn(g,n);
for i = 1:n
    Aini(:,i) = Aini(:,i)/norm(D_base*Aini(:,i));
end
Dict = D_base*Aini;
SnRdB = -10;

DictR.D_base = D_base;
DictR.Dict = Dict;          % Original Dictionary
DictR.iternum = noIt;
DictR.Tdata = K;
DictR.alpha = [0.2,0.02];     %for Algo1, Algo2
DictR.Tdict = 1;

%% Signal Preparation
% sigma = 0.31;
A = zeros(10,10); B = A; C = A;
A (2:6,2:6) = 1;    A1 = reshape(A,100,1);
B (8:9,8:9) = 1;    B1 = reshape(B,100,1);
C (5:9,5:9) = 1;    C1 = reshape(C,100,1);
T1 =  D_base(:,2)';                      %cos(2*pi*2.25*linspace(0,1,120)); %   
T2 =  D_base(:,4)';                      %cos(2*pi*1.5*linspace(0,1,120));
[Corr_T1,Corr_T2,Corr_A,Corr_B,Corr_C] = deal(zeros(Trials,5));

%% Event Generation No overlap a(A,B), partial overlap b(A,C), full overlap c(B,C)
% Choose the event from here
Ea = [A1,B1]';  %% setup event here
Ta = [T1;T2]';
Ya = (Ta*Ea);
PP = [1,2,3,4,5]; %[1,2,3,4,5,6,7];       % Which algos to display + calculate correlation
for tr = 1:Trials
    % Ya_n = Ya + sigma.*randn(size(Ya));
    Ya_n = awgn(Ya,SnRdB,'measured');

    %% Dictionary Learning Stuff
    
    [D(:,:,1),X(:,:,1)] = K_SVD(Ya_n,Dict,noIt,K);       % 
    [D(:,:,2),X(:,:,2)] = Sig_Rec(DictR,Ya_n,'ksvds');
    [D(:,:,3),X(:,:,3)] = S_1(Ya_n,Dict,noIt,K,0.2); 
    [D(:,:,4),X(:,:,4)] = Sig_Rec(DictR,Ya_n,'algo-1');
    [D(:,:,5),X(:,:,5)] = Sig_Rec(DictR,Ya_n,'algo-2');


    %% Recovery
%     for i = PP
%     A_ = abs(reshape(zscore(X(1,:,i)),10,10));
%     B_ = abs(reshape(zscore(X(2,:,i)),10,10));
% %     C_ = abs(reshape(X(3,:,i),10,10));
% 
%     % Plots
%     d = D(:,:,i);   % Recovered Time Series Normalization to Max = 1 & Correct the atom polarity
%     D(:,:,i) = d * diag(1./max(abs(d))) * diag([sign(d(1,1)),sign(d(1,2))]); %,sign(d(1,3))]);  
%     figure(i)   
%     subplot(n,3,1); imagesc((A_)); title('W_1'); set(gca,'YTick',[]); set(gca,'XTick',[]);
%     subplot(n,3,2:3); plot(D(:,1,i),'LineWidth',1.5);  title('D_1'); axis tight; colormap(gray); set(gca,'YTick',[]); set(gca,'XTick',[]);  
% 
%     subplot(n,3,4); imagesc((B_)); title('W_2'); set(gca,'YTick',[]); set(gca,'XTick',[]);
%     subplot(n,3,5:6); plot(D(:,2,i),'LineWidth',1.5);  title('D_2'); axis tight; set(gca,'YTick',[]); set(gca,'XTick',[]);
% 
% %     subplot(n,3,7); imagesc(C_); title('W_3');
% %     subplot(n,3,8:9); plot(D(:,3,i),'LineWidth',1.5);  title('D_3'); axis tight
% 
%     end

    %% Correlation stuff for Time Courses
    for k = PP
       d = D(:,:,k);   % Recovered Time Series Normalization to Max = 1 & Correct the atom polarity
       D(:,:,k) = d * diag(1./max(abs(d))) * diag([sign(d(1,1)),sign(d(2,2))]);%,sign(d(3,3))]); 
       Corr_T1(tr,k) =  max(corr(T1',D(:,:,k)));
       Corr_T2(tr,k) =  max(corr(T2',D(:,:,k)));   
    end
    
    %% Correlation stuff for Spatial Maps
    for k = PP
%        x = X(:,:,k);   % Recovered Time Series Normalization to Max = 1 & Correct the atom polarity
%        X(:,:,k) = x * diag(1./max(abs(x))) * diag([sign(x(1,1)),sign(x(2,2))]);%,sign(d(3,3))]); 
       Corr_A(tr,k) =  max(abs(corr(A1,X(:,:,k)')));
       Corr_B(tr,k) =  max(abs(corr(B1,X(:,:,k)')));   
       Corr_C(tr,k) =  max(abs(corr(C1,X(:,:,k)')));   
    end
end


%% Correlation results
fprintf('----------------------------------------------------------------\n');
fprintf('Results for \t\t\t KSVD \t KSVDs \t  S1 \t  A1 \t  A2\n');
fprintf('----------------------------------------------------------------\n');
fprintf('Mean Correlation of A: \t%0.4f,\t%0.4f,\t%0.4f,\t%0.4f,\t%0.4f\n',mean(Corr_A));
fprintf('Mean Correlation of B: \t%0.4f,\t%0.4f,\t%0.4f,\t%0.4f,\t%0.4f\n',mean(Corr_B));
fprintf('Mean Correlation of C: \t%0.4f,\t%0.4f,\t%0.4f,\t%0.4f,\t%0.4f\n\n',mean(Corr_C));

fprintf('Mean Correlation of T1: %0.4f,\t%0.4f,\t%0.4f,\t%0.4f,\t%0.4f\n',mean(Corr_T1));
fprintf('Mean Correlation of T2: %0.4f,\t%0.4f,\t%0.4f,\t%0.4f,\t%0.4f\n',mean(Corr_T2));
fprintf('----------------------------------------------------------------\n');
toc