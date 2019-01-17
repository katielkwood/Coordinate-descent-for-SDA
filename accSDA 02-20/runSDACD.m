b0 = ones(136,1);
X=train(:,2:137);
Y=zeros(23,2);
labels=train(:,1);
for i=1:23 
    Y(i, labels(i))=1;
end
Om = eye(136);
gam = 10^(-5);
lam = 10^(-5);
q = 50;
maxits = 100;
tol = 10^(-6);

[B, Q] = SDACD(b0, Y, X, Om, gam, lam, q, maxits, tol);