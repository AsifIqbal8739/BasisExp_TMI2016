%% S_1 for DIctionary Recovery
function [Count] = S_1(Y1,D_,D,noIt,s,alpha)
% D_ = Dummy Dictionary
% D = Original Dictionary ; Ground Truth

[n,K] = size(D_);
D_ = D_*diag(1./sqrt(sum(D_.*D_)));
Count = zeros(1,noIt);
for it = 1:noIt    
    G = D_'*D_;    
    W = omp(D_'*Y1,G,s);  
    for i = 1:K 
        D_i = removerows(D_',i);
        D_i = D_i';
        X_i = removerows(W,i);       
        E_k = Y1 - D_i*X_i;   
        d_k = D_(:,i);
        
        for j = 1:3
            W(i,:) = sign(d_k'*E_k).*(abs(d_k'*E_k)>=alpha/2).*(abs(d_k'*E_k)-alpha/2);
            d_k = E_k*W(i,:)'/norm(E_k*W(i,:)');
        end
		D_(:,i) = d_k;
    end    
    Count(1,it) = NumAtomRec(D_,D);               % To find # of atoms recovered
    disp(['S_1 Iteration # ',num2str(it),' Atoms recovered = ',num2str(Count(1,it))])
end
end


%% Function to Find # of Recovered Atoms
function Count = NumAtomRec(D_,D)
Count = 0;
% D_ = sign(D_).*D_;             % Making all atom values positive for safety
% D = sign(D).*D;
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