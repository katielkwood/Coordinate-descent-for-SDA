function [x] = test_prox_EN(A, d, x0, lam, alpha, maxits, tol)

% Applies proximal gradient algorithm to the l1-regularized quad
%   f(x) + g(x) = 0.5*x'*A*x - d'*x + lam*l1(x).
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% A: n by n positive definite coefficient matrix
% d: n dim coefficient vector.
% lam > 0: regularization parameter for l1 penalty.
% alpha: step length.
% maxits: number of iterations to run prox grad alg.
% tol: stopping tolerance for prox grad algorithm.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% x: solution at termination.

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initialization.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initial solution x.
x = x0;

% Get number of components of x,d, row/cols of A.
n = length(x);
Dx = 1;
its = 0;
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Outer loop: Repeat until converged or max # of iterations reached.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
while (Dx > tol && its < maxits)    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Check for convergence.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Compute gradient of differentiable part (f(x) = 0.5*x'*A*x - d'*x)
    df = A*x + d;
    
    x_old = x;
    % Update x using soft-thresholding.
    x = sign(x - alpha*df).*max(abs(x - alpha*df) - lam*alpha*ones(n,1), zeros(n,1));

    %Check convergence
    Dx = norm(x-x_old)/norm(x);
    
    
    
end

end