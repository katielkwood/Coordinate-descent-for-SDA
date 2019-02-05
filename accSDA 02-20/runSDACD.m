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

q = 3;
maxits = 100;
tol = 10^(-6);

[B, Q] = SDACD(b0, Y, X, Om, gam, lam, q, maxits, tol);

% Initialize.
K=4;
classMeans = zeros(p,K);

classes = [1; 2; 3; 4];
for i=1:K    
    % Extract training observations in class i.
    class_obs=X(classes==labels(i),:); 
    
    %Get the number of obs in that class (the number of rows)
    ni=size(class_obs,1);
    
    %Compute within-class mean
    classMeans(:,i)=mean(class_obs);
end

[stats,preds,proj,cent] = predict(B, test, classMeans);