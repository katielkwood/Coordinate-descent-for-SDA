function [B, Q, best_ind] = SDAP_val(train, val, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat)

% Applies accelerated proximal gradient algorithm with validation
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% train,val.Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% train,val.X: n by p data matrix.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lams > 0: vector of regularization parameters for l1 penalty.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
%
% feat: maximum fraction of nonzero features desired in validation scheme.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.
% best_ind: index of best solution in [B,Q].

%% Initialization.

% Extract X and Y from train.
X = train.X;
Y = train.Y;

% Get dimensions of input matrices.
[n, p] = size(X);
[~, K] = size(Y);

% Centroid matrix of training data.
C = diag(1./diag(Y'*Y))*Y'*X;

% Number of validation observations.
[nval,~] = size(val.X);

% Precompute repeatedly used matrix products
A = (X'*X + gam*Om); % Elastic net coefficient matrix.
alpha = 1/norm(A); % Step length in PGA.
D = 1/n*(Y'*Y); %D 
%XY = X'*Y; % X'Y.
R = chol(D);

% Initialize B and Q.
Q = ones(K,q);
B = zeros(p, q);



%% Validation Loop.

% Get number of parameters to test.
nlam = length(lams);

% Initialize validation scores.
val_scores = zeros(nlam, 1);

% Position of best solution.
best_ind = 1;

% Misclassification rate for each classifier.
mc = zeros(nlam, 1);

% Initialize B and Q.
Q = ones(K,q, nlam);
B = zeros(p, q, nlam);



%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Loop through potential regularization parameters.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
for ll = 1:nlam
    
    %%
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Call Alternating Direction Method to solve SDA.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    % For j=1,2,..., q compute the SDA pair (theta_j, beta_j).
    for j = 1:q
        
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Initialization.
        %+++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        % Compute Qj (K by j, first j-1 scoring vectors, all-ones last col).
        Qj = Q(:, 1:j, ll);
        
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
            d = 2*X'*(Y*(theta/n));
            
            % Update beta using proximal gradient step.
            b_old = beta;
            %tic
            [beta, ~] = prox_EN(A, d, beta, lams(ll), alpha, PGsteps, PGtol);
            %update_time = toc;
            
            % Update theta using the projected solution.
            % theta = Mj*D^{-1}*Y'*X*beta.
            b = Y'*(X*beta);
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
        Q(:,j, ll) = theta;
        B(:,j, ll) = beta;
    end
    
    %%
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Get classification statistics for (Q,B).
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    % Project validation data.
    % Project test data.
    PXtest = val.X*B(:,:, ll);
    % Project centroids.
    PC = C*B(:,:, ll);
    
    % Compute distances to the centroid for each projected test observation.
    dist = zeros(nval, K);
    for i = 1:nval
        for j = 1:K
            dist(i,j) = norm(PXtest(i,:) - PC(j,:));
        end
    end
    
    
    % Label test observation according to the closest centroid to its projection.
    [~,predicted_labels] = min(dist, [], 2);
    
    % Form predicted Y.
    Ypred = zeros(nval, K);
    for i=1:nval
        Ypred(i, predicted_labels(i)) = 1;
    end
    
    % Fraction misclassified.
    mc(ll) = (1/2*norm(val.Y - Ypred, 'fro')^2)/nval;  
    
    %%
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Validation scores.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    if nnz(B(:,:, ll)) <= (K-1)*p*feat % if fraction nonzero features less than feat.
        % Use misclassification rate as validation score.
        val_scores(ll) = mc(ll);
    else % Solution is not sparse enough, use most sparse as measure of quality instead.
        val_scores(ll) = nnz(B(:,:, ll));
    end
    
    % Update best so far.
    if (val_scores(ll) <= val_scores(best_ind))
        best_ind = ll;
    end
    
    % Display iteration stats.
    %if (quiet ==0)
        fprintf('ll: %d | lam: %1.5e| feat: %1.5e | mc: %1.5e | score: %1.5e | best: %d\n', ll, lams(ll), nnz(B(:,:,ll))/((K-1)*p), mc(ll),val_scores(ll), best_ind)
    %end
    
    
    
    
    
    
    
end % For ll = 1:nlam.

end % Function.

    
    
