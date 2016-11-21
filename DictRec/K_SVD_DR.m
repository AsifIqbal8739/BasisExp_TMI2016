%% KSVD implementation for Elad 2006 vDictionary Recovery
function [Count] = K_SVD_DR(Y1,D_,D,noIt,s)
% D_ = Dummy Dictionary
% D = Original Dictionary ; Ground Truth

[n,K] = size(D_);
D_ = D_*diag(1./sqrt(sum(D_.*D_)));
% D_ = D_.*repmat(sign(D_(1,:)),size(D_,1),1); % multiply in the sign of the first element.
Count = zeros(1,noIt);

for it = 1:noIt         
    W = omp(D_'*Y1,D_'*D_,s);
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
    Count(1,it) = NumAtomRec(D_,D);               % To find # of atoms recovered
    disp(['K-SVD Iteration # ',num2str(it),' Atoms recovered = ',num2str(Count(1,it))])
end
end

%% Function to Find # of Recovered Atoms
function Count = NumAtomRec(D_,D)
Count = 0;
for i = 1:size(D_,2)
    D_(:,i) = sign(D_(1,i))*D_(:,i);
end

for i = 1:size(D,2)
    d = sign(D(1,i))*D(:,i);
    dist = sum((D_ - repmat(d,1,size(D_,2))).^2);
    [~,Ind] = min (dist);
    Error = 1 - abs(D_(:,Ind)'*d);
    if Error < 0.01
        Count = Count + 1;
    end
end
Count = 100*Count/size(D_,2);
end