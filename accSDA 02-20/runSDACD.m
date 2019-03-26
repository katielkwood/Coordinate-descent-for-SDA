[n, p1] = size(train);
b0 = zeros(p1 - 1,1);
X=train(:,2:p1);
[n, p] = size(X);

K = 2;
Y=zeros(n,K);
labels=train(:,1);
for i=1:n 
    Y(i, labels(i))=1;
end

%% 
Om = eye(p);
gam = 10^(-5);
%gam = 10^(-1);
%lam = 10^(-5);
lam = 10^(-1);
mu = 10^(-5);

q = 1;
maxits = 1000;
Tol = 10^(-3);

%tol = struct('abs', 1e-4, 'rel', 1e-4);
tol.abs = 1e-4;
tol.rel = 1e-4;

PGsteps = 100;
PGtol = 10^(-6);

tic
[B_CD, Q_CD] = SDACD(b0, Y, X, Om, gam, lam, q, maxits, Tol);
CDtime = toc
%[B, Q] = SDAD(X, Y, Om, gam, lam, mu, q, PGsteps, PGtol, maxits, tol);
tic
[B_AP,Q_AP] = SDAAP(X, Y, Om, gam/4, lam/4, q, PGsteps, PGtol, maxits, Tol);
APtime = toc

% Initialize.
classMeans = zeros(p,K);

classes = train(:,1);
labels = unique(classes);

%%%
for i=1:K    
    % Extract training observations in class i.
    class_obs=X(classes==labels(i),:); 
    
    %Get the number of obs in that class (the number of rows)
    ni=size(class_obs,1);
    
    %Compute within-class mean
    classMeans(:,i)=mean(class_obs);
end

[statsCD,predsCD,projCD,centCD] = predict(B_CD, test, classMeans);

[statsAP,predsAP,projAP,centAP] = predict(B_AP, test, classMeans);

%check constraints
normalizedConstr = Q_CD'*Y'*Y*Q_CD

statsCD
statsAP
