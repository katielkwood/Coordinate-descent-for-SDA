function [B, Q, lbest, lambest] = SDAAPcv(train, folds, Om, gam, lams, q, PGsteps, PGtol, maxits, tol, feat, quiet)

% Applies accelerated proximal gradient algorithm with cross validation
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% train,val.Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% folds: number of folds to use in K-fold cross-validation.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.
% gam > 0: regularization parameter for elastic net penalty.
% lams > 0: vector of regularization parameters for l1 penalty.
% q: desired number of discriminant vectors.
% PGsteps: max its of inner prox-grad algorithm to update beta.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
% feat: maximum fraction of nonzero features desired in validation scheme.
% quiet: toggles display of iteration stats.
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Output
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% B: p by q by nlam matrix of discriminant vectors.
% Q: K by q by nlam matrix of scoring vectors.
% best_ind: index of best solution in [B,Q].


%% Initialize training sets, etc.

% Extract X and Y from train.
X = train.X;
Y = train.Y;

% Get dimensions of input matrices.
[n, p] = size(X);
[~, K] = size(Y);

% If n is not divisible by K, duplicate some records for the sake of
% cross validation.
pad = 0; % Initialize number of padding observations.
if mod(n,folds) > 0
    % number of elements to duplicate.
    pad = ceil(n/folds)*folds - n;

    % duplicate elements of X and Y.
    X = [X; X(1:pad, :)];
    Y = [Y; Y(1:pad, :)];
end

% Get new size of X.
[n, ~] = size(X);

% Randomly permute rows of X.
prm = randperm(n);
X = X(prm, :);
Y = Y(prm, :);

% Sort lambdas in descending order (break ties by using largest lambda =
% sparsest vector).
lams = sort(lams, 'descend');


%% Initialize cross-validation indices.

% Number of validation samples.
nv = n/folds;

% Initial validation indices.
vinds = (1:nv)';

% Initial training indices.
tinds = ((nv+1):n)';

% Get number of parameters to test.
nlam = length(lams);

% Validation scores.
scores = q*p*ones(folds, nlam);

% Misclassification rate for each classifier.
mc = zeros(folds, nlam);



for f = 1 : folds

    %% Initialization.


    % Extract X and Y from train.
    Xt = X(tinds, :);
    Yt = Y(tinds, :);

    % Extract training data.
    Xv = X(vinds, :);
    Yv = Y(vinds, :);
    % Get dimensions of training matrices.
    [nt, p] = size(Xt);

    % Centroid matrix of training data.
    C = diag(1./diag(Yt'*Yt))*Yt'*Xt;


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


    %% Validation Loop.

    if (quiet ==0)
        fprintf('++++++++++++++++++++++++++++++++++++\n')
        fprintf('Fold %d \n', f)
        fprintf('++++++++++++++++++++++++++++++++++++\n')
    end
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Loop through potential regularization parameters.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for ll = 1:nlam

        % Initialize B and Q.
        Q = ones(K,q);
        B = zeros(p, q);

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Call Alternating Direction Method to solve SDA.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % For j=1,2,..., q compute the SDA pair (theta_j, beta_j).
        %[f, lams(ll)]
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
                d = 2*Xt'*(Yt*(theta/nt));

                % Update beta using proximal gradient step.
                b_old = beta;
                %tic
                [beta, ~] = APG_EN2(A, d, beta, lams(ll), alpha, PGsteps, PGtol);
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

                % Check convergence.
                if max(db, dt) < tol
                    % Converged.
                    %fprintf('Algorithm converged for %g iterations\n', j)
                    break
                end
            end

            % Update Q and B.
            Q(:,j) = theta;
            B(:,j) = beta;
        end

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Get classification statistics for (Q,B).
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        % Project validation data.
        PXtest = Xv*B(:,:);
        % Project centroids.
        PC = C*B(:,:);

        % Compute distances to the centroid for each projected test observation.
        dist = zeros(nv, K);
        for i = 1:nv
            for j = 1:K
                dist(i,j) = norm(PXtest(i,:) - PC(j,:));
            end
        end


        % Label test observation according to the closest centroid to its projection.
        [~,predicted_labels] = min(dist, [], 2);

        % Form predicted Y.
        Ypred = zeros(nv, K);
        for i=1:nv
            Ypred(i, predicted_labels(i)) = 1;
        end

        % Fraction misclassified.
        mc(f, ll) = (1/2*norm(Yv - Ypred, 'fro')^2)/nv;

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Validation scores.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        %nnz(B)
        if 1<= nnz(B) && nnz(B) <= q*p*feat % if fraction nonzero features less than feat.
            % Use misclassification rate as validation score.
            scores(f, ll) = mc(f, ll);
%         elseif nnz(B) < 0.5; % Found trivial solution.
%             %fprintf('dq \n')
%             scores(f, 11) = 10000; % Disqualify with maximum possible score.
        elseif nnz(B) > q*p*feat % Solution is not sparse enough, use most sparse as measure of quality instead.
            scores(f, ll) = nnz(B);
        end


        % Display iteration stats.
        if (quiet ==0)
            fprintf('f: %3d | ll: %3d | lam: %1.5e| feat: %1.5e | mc: %1.5e | score: %1.5e \n', f,  ll, lams(ll), nnz(B(:,:))/((K-1)*p), mc(f, ll), scores(f, ll))
        end


    end % For ll = 1:nlam.

    %+++++++++++++++++++++++++++++++++++
    % Update training/validation split.
    %+++++++++++++++++++++++++++++++++++
    % Extract next validation indices.
    tmp = tinds(1:nv);

    % Update training indices.
    tinds = [tinds((nv+1):nt); vinds];

    % Update validation indices.
    vinds = tmp;

end % folds.

%%  Find best solution.

% average CV scores.
avg_score = mean(scores);

% choose lambda with best average score.
[~, lbest] = min(avg_score);

lambest = lams(lbest);


%% Solve with lambda = lam(lbest).

% Use full training set.
Xt = X;
Yt = Y;

[B,Q] = SDAAP(Xt, Yt, Om, gam, lams(lbest), q, PGsteps, PGtol, maxits, tol);
