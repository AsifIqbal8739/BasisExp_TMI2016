% Code for Basis function of gamma variates of different frequencies
% D = basis matrix with columns of gamma variates of different frequencies
% Size(D) = [m,n];
function D = basis_gamma(m,n)
D = zeros(m,n);
D(:,1) = 1;

GG = spm_Gpdf(0:100,10,0.48);   % Manual tuning for the first basis vector!!!!
% figure;plot(GG)
St = GG(1:50);
[a,b] = rat(50/m);
St_ = decimate(interp(St,b),a);
% figure(); subplot(4,5,1); plot(D(:,1),'linewidth',2.5);
for i =1:n-1
    if i == 1
        temp = St(1:m);
        tt = 1.0;
    else        
        [a,b] = rat(tt);
        temp = decimate(interp(St_,b*100),a*100);
        temp = repmat(temp,1,i);
        temp = temp(1:m);
        tt = tt + 0.5;
    end
    D(:,i+1) = temp';
%     plot(temp);
%     if i < 20
%         subplot(4,5,i+1); plot(D(:,i+1),'linewidth',2.5); axis tight;
%         title(tt);
%     end
end
    D = normc(D);
end