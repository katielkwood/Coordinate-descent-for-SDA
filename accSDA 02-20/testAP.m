function [B,Q] = testAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol)
% Applies accelerated proximal gradient algorithm 
% to the optimal scoring formulation of
% sparse discriminant analysis in the form used in the paper Proximal
% Methods for Sparse Optimal Scoring and Discriminant Analysis by Ames,
% Atkins, Einarsson, and Clemmensen
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Xt: n by p data matrix.
% Yt: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lam > 0: regularization parameters for l1 penalty.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% PGtol: stopping tolerance for inner APG method.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.


% Read training data size.
[n, p] = size(Xt);
[~, K] = size(Yt);

% Precompute repeatedly used matrix products
A = 2*(Xt'*Xt + gam*Om); % Elastic net coefficient matrix.
alpha = 1/norm(A); % Step length in PGA.
D = (1/n)*(Yt'*Yt); %D 
%XY = X'*Y; % X'Y.
R = chol(D);

% Initialize B and Q.
Q = ones(K,q);
B = zeros(p, q);

%+++++++++++++++++++++++++++++++++++++++++++++++++++++
% Alternating direction method to update (theta, beta)
%+++++++++++++++++++++++++++++++++++++++++++++++++++++
for j = 1:q
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Initialization.
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Compute Qj (K by j, first j-1 scoring vectors, all-ones last col).
    Qj = Q(:, 1:j);
    
    % Precompute Mj = I - Qj*Qj'*D.
    Mj = @(u) u - Qj*(Qj'*(D*u));
    
    % Initialize theta.
    theta = Mj(rand(K,1));
    theta = theta/sqrt(theta'*D*theta);
    
    % Initialize beta.
    beta = zeros(p,1);
    
    
    
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Alternating direction method to update (theta, beta)
    %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    for its = 1:maxits
        
        % Compute coefficient vector for elastic net step.
        d = -2*Xt'*(Yt*(theta));
        
        % Update beta using proximal gradient step.
        b_old = beta;
        %tic
        [beta] = test_prox_EN(A, d, beta, lam, alpha, PGsteps, PGtol);
        %update_time = toc;
        
        % Update theta using the projected solution.
        % theta = Mj*D^{-1}*Y'*X*beta.
        b = Yt'*(Xt*beta);
        y = R'\b;
        z = R\y;
        tt = Mj(z);
        t_old = theta;
        theta = tt/sqrt(tt'*D*tt);
        
        % Progress.
        db = norm(beta-b_old)/norm(beta);
        dt = norm(theta-t_old)/norm(theta);
        %fprintf('It %5.0f      db %5.2f      dt %5.2f      Subprob its %5.0f It time %5.2f\n', its, db, dt, subprob_its, update_time)
        
        % Check convergence.
        if max(db, dt) < tol
            % Converged.
            %fprintf('Algorithm converged after %g iterations\n\n', its)
            break
        end
    end
    
    % Update Q and B.
    Q(:,j) = theta;
    B(:,j) = beta;
end
    

