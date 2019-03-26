
function [B, Q] = SDACD(b0, Y, X, Om, gam, lam, q, maxits, tol)

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

% Initialize B and Q and Beta.
Q = ones(K,q);
B = zeros(p, q);
Beta = b0;

A = 2*(X'*X + gam*Om); 
%-------------------------------------------------------------------
% Matrices for theta update.
%-------------------------------------------------------------------
D = (1/n)*(Y'*Y);
R = chol(D); % Cholesky factorization of D.
%---------------------------------------
%-------------------------------------------------------------------
% Alpha. Change later.
%-------------------------------------------------------------------
%A = X'*X;
%alpha = 1/norm(A);
alpha = 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outer loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% For j=1,2,..., q compute the SDA pair (theta_j, Beta_j).
db = 1;
dt = 1;
for j = 1:q
    SDACDits = 0;

     %+++++++++++++++++++++++++++++++++++++++++++++++++++++
     % Initialization.
     %+++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Compute Qj (K by j, first j-1 scoring vectors, all-ones last col).
    Qj = Q(:, 1:j);
    
    % Precompute Mj = I - Qj*Qj'*D.
    Mj = @(u) u - Qj*(Qj'*D*u);
    
    %compute D^-1*Y^T*X*Beta
    y = R'\(Y'*X*Beta);
    z = R\y;
    
    % Initialize theta.
    theta = Mj(rand(K,1)); 
    theta = theta/sqrt(theta'*D*theta);
   
    % Initialize coefficient vector for elastic net step.
    d = 2*X'*(Y*(theta));  %line 199 in Ames paper
                        %should this be negative?
    

    while (max(db, dt) > tol && SDACDits < maxits)

        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Coordinate descent method for updating (theta, Beta)
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        b_old = Beta;
    
        Beta = betaCoordDesc(Beta, j, alpha, X, Y, theta, Q, Om, lam, gam, maxits, tol, d, A);
    
        normBeta = norm(Beta);
    
        % Update theta using the projected solution.
    
        b = Y'*(X*Beta);
        y = R'\b;
        z = R\y;
        tt = Mj(z);
        t_old = theta;
        theta = tt/sqrt(tt'*D*tt);
            
        % Update changes.
        db = norm(Beta-b_old)/norm(Beta);
        dt = norm(theta-t_old)/norm(theta);
        
        %fprintf('it: %d | db: %1.3e | dt: %1.3e | tol: %1.3e \n', SDACDits, db, dt, tol)
        
    
        SDACDits = SDACDits + 1; 
    end
    if SDACDits == maxits 
        fprintf('Algorithm did not converge')
        %break
    end
    % Update Q and B.
    Q(:,j) = theta;
    B(:,j) = Beta;
    SDACDits; 
    
 
end

end
