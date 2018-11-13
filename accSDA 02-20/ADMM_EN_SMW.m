function [x,y,z, k] = ADMM_EN_SMW(Ainv, V,R, d, x0, lam, mu, maxits, tol, quiet)

% Applies Alternating Direction Method of Multipliers to the l1-regularized
% quadratic program
%   f(x) + g(x) = 0.5*x'*A*x - d'*x + lam*l1(x).
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% A: n by n positive definite coefficient matrix
% Ainv: diagonal of A^{-1} term in SMW formula.
% VAinv: V*A^{-1} = (A^{-1}U)' product in SMW formula.
% R: upper triangular matrix in Chol decomp of I + U*Ainv*V.
% d: n dim coefficient vector.
% lam > 0: regularization parameter for l1 penalty.
% mu > 0: augmented Lagrangian penalty parameter.
% alpha: step length.
% maxits: number of iterations to run.
% tol = [tol.abs, tol.rel]: stopping tolerances.
% quiet = controls display of it'n statistics.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% (x, y,z): solution at termination.
% k: number of iterations needed.

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initialization.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Initial solutions.
x = x0;
y = x0;
p = length(x);
z = zeros(p,1);
[n,~] = size(V);

if quiet==0
    display('---------------------------------------------------------------')    
%     xb = (R'*R - mu*eye(p))\d;    
%     lam_max = (d'*xb - 0.5*(R*xb)'*(R*xb));
%     fprintf('lam_max %g || l1 %g || Ratio %g\n', lam_max, norm(xb,1)/norm(xb), lam_max*norm(xb)/norm(xb,1));
%     x = xb;
%     y = xb;
%     display('---------------------------------------------------------------')
end
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Outer loop: Repeat until converged or max # of iterations reached.
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for k = 0:maxits
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Update x using SMW applied to
    % (mu I + gam*Om + X'X)x = d + mu*y - z.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % RHS coefficient vectors.
    b = d + mu*y-z;    
    btmp = V*(Ainv.*b)/n;
    
    % Apply SMW to get x.
    x = Ainv.*b - 2*Ainv.*(V'*(R\(R'\btmp)));
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Update y.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Update y using soft-thresholding.
    yold = y;
    tmp = x + z/mu;
    y = sign(tmp).*max(abs(tmp) - lam*ones(p,1), zeros(p,1));
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Update z.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    z = z + mu*(x-y);
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Check for convergence.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    %%% Primal constraint violation.
    
    % Primal residual.
    r = x - y;
    
    % l2 norm of the residual.
    dr = norm(r);
    
    %%% Dual constraint violation.
    
    % Dual residual.
    s = mu*(y - yold);
    
    % l2 norm of the residual.
    ds = norm(s);
    
    %%%  Check if the stopping criteria are satisfied.
    
    % Compute absolute and relative tolerances.
    ep = sqrt(p)*tol.abs + tol.rel*max(norm(x), norm(y));
    es = sqrt(p)*tol.abs + tol.rel*norm(y);
    
    % Display current iteration stats.
    if (quiet==0)
        fprintf('it = %g, primal_viol = %3.2e, dual_viol = %3.2e, norm_y = %3.2e\n', k, dr-ep, ds-es, max(norm(x), norm(y)))
    end
    
    % Check if the residual norms are less than the given tolerance.
    if (dr < ep && ds < es)
        break % The algorithm has converged.
    end
end

if quiet==0
    display('---------------------------------------------------------------')
end