%% KSVD implementation for Practice
function [D_,W] = K_SVD(Y1,D_,noIt,s)

K = size(D_,2);
D_ = D_*diag(1./sqrt(sum(D_.*D_)));

for it = 1:noIt          
    G = D_'*D_;
    W = omp(D_'*Y1,G,s);
    
    %% K-SVD
    R = Y1 - D_*W;
    for k=1:K
        I = find(W(k,:));
        Ri = R(:,I) + D_(:,k)*W(k,I);
        [U,S,V] = svds(Ri,1,'L');
%         U is normalized
        D_(:,k) = U;
        W(k,I) = S*V';

        R(:,I) = Ri - D_(:,k)*W(k,I);
    end    
%     disp(['Total Error in K-SVD = ',num2str(norm(Y1 - D_*W))])
    
end
W = full(W);
end
