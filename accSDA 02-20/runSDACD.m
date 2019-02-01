[n, p1] = size(train);
b0 = ones(p1 - 1,1);
X=train(:,2:p1);
[n, p] = size(X);
Y=zeros(n,2);
labels=train(:,1);
for i=1:n 
    Y(i, labels(i))=1;
end

%% 
Om = eye(p);
gam = 10^(-5);
%gam = 10^(-1);
%lam = 10^(-5);
lam = gam

q = 5;
maxits = 100;
tol = 10^(-6);

[B, Q, Beta] = SDACD(b0, Y, X, Om, gam, lam, q, maxits, tol);