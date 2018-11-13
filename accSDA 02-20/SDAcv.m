function [B, Q, lbest] = SDAcv(train, folds, gam, feats, q, maxits, tol, quiet)

% Applies proximal gradient algorithm with cross validation
% to the optimal scoring formulation of
% sparse discriminant analysis proposed by Clemmensen et al. 2011.
%
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Input
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% train.X: n by p data matrix.
% train.Y: n by K matrix of indicator variables (Yij = 1 if i in classs j)
% folds: number of folds to use in K-fold cross-validation.
% Om: p by p parameter matrix Omega in generalized elastic net penalty.%
% gam: fixed ridge regression penalty parameter.
% feat: vector of desired number of nonzero features.
% q: desired number of discriminant vectors.
% maxits: number of iterations to run alternating direction alg.
% tol: stopping tolerance for alternating direction algorithm.
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

% Recompute size.
[n, ~] = size(X);

% Randomly permute rows of X.
prm = randperm(n);
X = X(prm, :);
Y = Y(prm, :);

% Sort lambdas in descending order (break ties by using largest lambda =
% sparsest vector).
lams = sort(feats, 'descend');


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

% Maximum fraction of desired nonzero features.
feat = max(feats)/(q*p);

for f = 1 : folds

    %% Initialization.

    % Extract X and Y from train.
    Xt = X(tinds, :);
    Yt = Y(tinds, :);

    % Extract validation data.
    Xv = X(vinds, :);
    Yv = Y(vinds, :);
    % Get dimensions of training matrices.
    [nt, p] = size(Xt);

    % Centroid matrix of training data.
    C = diag(1./diag(Yt'*Yt))*Yt'*Xt;


%     % Precompute repeatedly used matrix products
%     A = (Xt'*Xt + gam*Om); % Elastic net coefficient matrix.
%     alpha = 1/norm(A); % Step length in PGA.
%     D = 1/n*(Yt'*Yt); %D
%     %XY = X'*Y; % X'Y.
%     R = chol(D);

    %% Validation Loop.

    if (quiet == 0)
        fprintf('++++++++++++++++++++++++++++++++++++\n')
        fprintf('Fold %d \n', f)
        fprintf('++++++++++++++++++++++++++++++++++++\n')
    end

    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Loop through potential regularization parameters.
    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for ll = 1:nlam

        %%
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        % Solve for B using training data.
        %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        [B, ~,] = slda(Xt, Yt, gam, feats(ll), q, maxits, tol, 0);

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
            fprintf('f: %3d | ll: %3d | lam: %1.5e| feat: %1.5e | mc: %1.5e | score: %1.5e \n', f,  ll, lams(ll), nnz(B(:,:))/(q*p), mc(f, ll), scores(f, ll))
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

%% Solve with lambda = lam(lbest).

% Finished training lambda.
%fprintf('Finished Training: lam = %d \n', lambest)



% Use full training set.
Xt = X(1:(n-pad), :);
Yt = Y(1:(n-pad), :);

% size(Xt)
% size(lams)
% lbest
% size(Yt)

[B,Q] = slda(Xt, Yt, gam, feats(lbest), q, maxits, tol, 0);

% fprintf('Found DVs\n')
