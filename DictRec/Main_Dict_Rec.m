%% Main Script for Dictionary Recovery using Double Sparsity Stuff
clc; clear all; close all;
rng('default')

%% Data Stuff
m = 20;     n = 50;    g = 20;     N = 1500;      % D(m,n), Y(m,N)
K = 2; %[2,3,4,5];        % Signal Sparsity Level
H = 10;     % Dictionary Sparsity Level
SnRdB = 20; %[5,10,20,35,50];
noIt = 11*K.^2;
Trials = 30;
[Count_K,Count_Ks,Count_A1,Count_A2,Count_S1] = deal(zeros(Trials,noIt)); 
Count_FIDL = zeros(1,Trials);

%% Dictionary Creation & Normalization
D_base = odctdict(m,g); 
% D_base = basis_gamma(m,g);
Aini = randn(g,n);
for i = 1:n
    p = randperm(g);    Aini(p(1:H),i) = 0; 
    Aini(:,i) = Aini(:,i)/norm(D_base*Aini(:,i));
end
Dict = D_base*Aini;
% [a, MSGID] = lastwarn()
warning('off','MATLAB:nearlySingularMatrix');        % turn off warning about bad-conditioned matrices

%% Parameter Setup 
DictR.D_base = D_base;
DictR.Dict = Dict;          % Original Dictionary
DictR.iternum = noIt;
DictR.Tdata = K;
DictR.alpha = [0.25,0.18,0.45];     %for Algo1, Algo2, and S_1
DictR.Tdict = 10;
% DictR.Method = 'ksvd';          % ksvd, ksvds, algo1, algo2
aa = DictR.alpha(3);
tic;
%% Loops and stuff
parfor tr = 1:Trials
    [~,~,Y1] = gererateNoiseAddedSyntheticData(N,K,Dict,SnRdB);
    disp(['K-SVD Started for Trial #',num2str(tr)]);
         [Count_K(tr,:)] = K_SVD_DR(Y1,D_base*randn(g,n),Dict,noIt,K);
    disp(['K-SVDs Started for Trial #',num2str(tr)]);
         [Count_Ks(tr,:)] = Dict_Rec(DictR,Y1,'ksvds');
     disp(['Algo_3 Started for Trial #',num2str(tr)]);
         [Count_A1(tr,:)] = Dict_Rec(DictR,Y1,'algo-1');
     disp(['Algo_4 Started for Trial #',num2str(tr)]);
         [Count_A2(tr,:)] = Dict_Rec(DictR,Y1,'algo-2');
    disp(['S1 Started for Trial #',num2str(tr)]);
         [Count_S1(tr,:)] = S_1(Y1,D_base*randn(g,n),Dict,noIt,K,aa);
%     Count_FIDL(tr) = FIDL(Y1,D_base*randn(g,n),Dict);
end
warning('on','MATLAB:nearlySingularMatrix');        % turn off warning about bad-conditioned matrices
% fprintf('FIDL: %0.2f\n',mean(Count_FIDL));
%% Recovery 
toc
disp(['K-SVD  Ratio: ',num2str(mean(Count_K(:,end)))]);
disp(['K-SVDs Ratio: ',num2str(mean(Count_Ks(:,end)))]);
disp(['Algo_1 Ratio: ',num2str(mean(Count_A1(:,end)))]);
disp(['Algo_2 Ratio: ',num2str(mean(Count_A2(:,end)))]);
disp(['S_1 Ratio: ',num2str(mean(Count_S1(:,end)))]);

%% Plots
% plot(mean(T.Count_K),'k-.','LineWidth',2)
% hold on
% plot(mean(T.Count_Ks),'g--','LineWidth',2)
% plot(mean(T.Count_A1),'r-.','LineWidth',2)
% plot(mean(T.Count_A2),'m','LineWidth',2)
% plot(mean(T.Count_S1),'b--','LineWidth',2)
% legend('K-SVD','K-SVDs','A_1','A_2','S_1');
% title(['SNR = ' num2str(SnRdB),' dB']);
% hold off
    


