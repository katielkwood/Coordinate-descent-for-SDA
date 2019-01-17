function [B, Q] = CoordDescent_SDA(x0, Y, X, Om, gam, lam, q, maxits, tol)

% Applies coordinate descent algorithm to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
% 
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% X: n by p data matrix.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lam > 0: regularization parameter for l1 penalty.
% q: desired number of discriminant vectors.
% maxits: number of iterations to run coordinate descent alg.
% tol: stopping tolerance for coordinate descent algorithm.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q matrix of discriminant vectors.
% Q: K by q matrix of scoring vectors.



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Get dimensions of input matrices.
[n, p] = size(X);
[~, K] = size(Y);

% Initialize B and Q.
Q = ones(K,q);
B = ones(p, q);

D = (1/n)*Y'*Y;
Dinv = inv(D) %I know we don't usually calculate the inverse in practice, but I don't know how else to implement it in calculating w

w = (eye(K) - Q(:,1)*Q(:,1)'*D)*Dinv*Y'*X*B(:,1);
Q(:,1) = w/sqrt(w'*D*w);

for t=1:q-1
    %are we updating theta first or beta?
    w = (eye(K) - Q(:,1:t)*Q(:,1:t)'*D)*Dinv*Y'*X*B(:,t);
    Q(:,t+1) = w/sqrt(w'*D*w);
    for j=1:p
        %this calculates B(:,2:q) but I don't know how to get the first
        %column
        B(j,t+1) = sign(2*X(:,j)'*(Y*Q(:,t+1)-Y*Q(:,t)))*max(abs(2*X(:,j)'*(Y*Q(:,t+1)-Y*Q(:,t))) - lam, 0)/(1 + 2*gam);
    end
end

end