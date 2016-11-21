%% S_1 for Practice
function [D_,W] = S_1(Y1,D_,noIt,s,alpha)
% D_ = Dummy Dictionary
% D = Original Dictionary ; Ground Truth

[n,K] = size(D_);
D_ = D_*diag(1./sqrt(sum(D_.*D_)));

for it = 1:noIt    
    G = D_'*D_;    
    W = full(omp(D_'*Y1,G,s));  
    for i = 1:K 
        D_i = removerows(D_',i);
        D_i = D_i';
        X_i = removerows(W,i);       
        E_k = Y1 - D_i*X_i;   
        d_k = D_(:,i);
        
        for j = 1:3
            W(i,:) = sign(d_k'*E_k).*(abs(d_k'*E_k)>=alpha/2).*(abs(d_k'*E_k)-alpha/2);
            D_(:,i) = E_k*W(i,:)'/norm(E_k*W(i,:)');
        end
    end       
end
end
