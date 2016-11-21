%% Code for FAST and Incoherent Dictionary Abol2015. FIDL

function Count = FIDL(Y,D,D_Orig)
kmax = 20;  rho = 10^-7;    mu = 0.1;   lambda = 0.5; epsilon = 0.001;
eta = 0.001;    it = 0;
D = normc(D);
X = randn(size(D,2),size(Y,2));
while norm(Y-D*X,'fro')>= epsilon
    for k = 1:kmax
        D = D - eta*(D*X*X' - Y*X' + 2*mu*D*(D'*D-eye(size(D,2))));
        D = normc(D);
    end
    beta = 1/norm(D'*D);
    for k = 1:kmax
        tt = X - 2*beta*D'*(D*X-Y);
        X = X.*max(0,(1-(beta*lambda)./(abs(tt))));
    end
    eta = 1/norm(X*X');
    Xtilde = X - 2*beta*D'*(D*X-Y);
    tt = 2*beta*trace(sign(Xtilde)'*D'*(Y-D*Xtilde)) + 2*beta^2*lambda*trace(sign(Xtilde)'*D'*D*sign(Xtilde)) ...
        + sum(sum(sign(Xtilde - beta*lambda*sign(Xtilde)).*sign(Xtilde - 2*beta*lambda*sign(Xtilde))));
    lambda = lambda - rho*tt;
%     if it < 30
        it = it+1;
%     else 
%         break;
%     end
    fprintf('Iteration: %d, error: %0.2f, lambda: %0.2f\n',it,norm(Y-D*X,'fro'),lambda);
    if lambda <=0; break; end
end

Count = NumAtomRec(D,D_Orig);
return;
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