function [B,Q] = SDAAP(Xt, Yt, Om, gam, lam, q, PGsteps, PGtol, maxits, tol)
% Applies accelerated proximal gradient algorithm 
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
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
[nt, p] = size(Xt);
[~, K] = size(Yt);

% Precompute repeatedly used matrix products
%display('form EN coefficient matrix')
%gamOm = gam*Om;
tic
if norm(diag(diag(Om)) - Om, 'fro') < 1e-15 % Omega is diagonal.
    A.flag = 1;
    % Store components of A.
    A.gom = gam*diag(Om);
    A.X = Xt;
    A.n = nt;
    
    alpha = 1/( 2*(norm(Xt,1)*norm(Xt,'inf')/nt + norm(A.gom, 'inf') ));
    %alpha = 1/( 2*(norm(X)^2/n + norm(A.gom, 'inf') ));
    %     A.A = 2*(X'*X/n + diag(A.gom));
    %     fprintf('Test bounds: L = %g, Lt = %g', norm(A.A), 2*(norm(X,1)*norm(X,'inf')/n + norm(gamOm, 'inf') ))
    %     norm(A.A)
else
    A.flag = 0;
    A.A = 2*(Xt'*Xt/nt + gam*Om); % Elastic net coefficient matrix.
    alpha = 1/norm(A.A, 'fro');
end
%Atime = toc;
%fprintf('Coefficient time %g\n', Atime);
%tic
%alpha = 1/norm(A.A); % Step length in PGA.
D = 1/nt*(Yt'*Yt); %D
%XY = X'*Y; % X'Y.
%other =toc;
%fprintf('Other preprocessing %g\n', other);
%tic;
R = chol(D);
%Rtime = toc;
%fprintf('Chol time %g\n', Rtime);


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
    
    
    
    for its = 1:maxits
        
        % Compute coefficient vector for elastic net step.
        d = 2*Xt'*(Yt*(theta/nt));
        
        % Update beta using proximal gradient step.
        b_old = beta;
        %tic
        [beta, ~] = APG_EN2(A, d, beta, lam, alpha, PGsteps, PGtol);
        %update_time = toc;
        
        % Update theta using the projected solution.
        % theta = Mj*D^{-1}*Y'*X*beta.        
        if norm(beta) > 1e-12
            % update theta.
            b = Yt'*(Xt*beta);
            y = R'\b;
            z = R\y;
            tt = Mj(z);
            t_old = theta;
            theta = tt/sqrt(tt'*D*tt);
            
            % Update changes..
            db = norm(beta-b_old)/norm(beta);
            dt = norm(theta-t_old)/norm(theta);
        
        else
            % Update b and theta.
            beta = 0;
            theta = 0;
            % Update change.
            db = 0;
            dt = 0;
        end
      
        
        % Progress.              
        %fprintf('It %5.0f      db %5.2f      dt %5.2f  \n', its, db, dt)
        
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

