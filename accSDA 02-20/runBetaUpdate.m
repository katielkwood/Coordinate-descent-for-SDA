[n, p1] = size(train);
b0 = zeros(p1 - 1,1);
X=train(:,2:p1);
[n, p] = size(X);

K=2;
Y=zeros(n,K);
labels=train(:,1);
for i=1:n 
    Y(i, labels(i))=1;
end

A = 2*(X'*X + gam*Om); 
Om = eye(p);
gam = 10^(-5);
lam = gam;
mu = 10^(-5);

q = 1;
j = 1;
alpha = 1;
maxits = 100;
Tol = 10^(-6);
Q = ones(K,q);
theta = [1;-1];

%tol = struct('abs', 1e-4, 'rel', 1e-4);
tol.abs = 1e-4;
tol.rel = 1e-4;

PGsteps = 100;
PGtol = 10^(-6);

d = 2*X'*(Y*(theta/n));

BetaCD = betaCoordDesc(b0, j, alpha, X, Y, theta, Q, Om, lam, gam, 10, 10^(-3));
[BetaAP, ~] = prox_EN(A, d, b0, lam, alpha, PGsteps, PGtol);

classMeans = zeros(p,K);

classes = train(:,1);
labels = unique(classes);

%%
for i=1:K    
    % Extract training observations in class i.
    class_obs=X(classes==labels(i),:); 
    
    %Get the number of obs in that class (the number of rows)
    ni=size(class_obs,1);
    
    %Compute within-class mean
    classMeans(:,i)=mean(class_obs);
end

[BstatsCD,BpredsCD,BprojCD,BcentCD] = predict(BetaCD, test, classMeans);
[BstatsAP,BpredsAP,BprojAP,BcentAP] = predict(BetaAP, test, classMeans);

BstatsCD
BstatsAP
