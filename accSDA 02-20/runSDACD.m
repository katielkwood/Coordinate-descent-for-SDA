[n, p1] = size(train);
b0 = zeros(p1 - 1,1);
X=train(:,2:p1);
[n, p] = size(X);

K = 4;
Y=zeros(n,K);
labels=train(:,1);
for i=1:n 
    Y(i, labels(i))=1;
end

%% 
Om = eye(p);
gam = 10^(-5);
lam = 10^(-1);

q = K-1;
maxits = 300;
Tol = 10^(-3);

PGsteps = 100;
PGtol = 10^(-3);

tic
[B_CD, Q_CD] = SDACD(b0, Y, X, Om, gam, lam, q, maxits, Tol);
CDtime = toc
%[B, Q] = SDAD(X, Y, Om, gam, lam, mu, q, PGsteps, PGtol, maxits, tol);
tic
%[B_AP,Q_AP] = SDAP(X, Y, Om, gam/4, lam/4, q, PGsteps, PGtol, maxits, Tol);
[testB_AP,testQ_AP] = testAP(X, Y, Om, gam, lam, q, PGsteps, PGtol, maxits, Tol);
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

[statsAP,predsAP,projAP,centAP] = predict(testB_AP, test, classMeans);

%check constraints
normalizedConstr = Q_CD'*Y'*Y*Q_CD

statsCD
statsAP

%plot([B_CD, testB_AP])
plot(B_CD)
