function [Beta] = betaCoordDesc(b0, j, alpha, X, Y, theta, Q, Om, lam, gam, maxits, tol, d, A)

[n, p] = size(X);
Db = 1;
b_old = b0;
Beta = b0;
betaIts = 0;
while (Db > tol && betaIts < maxits)
    %Beta = Beta/norm(Beta);
    b_old = Beta;
    
    for i = 1:length(Beta)
               
        %based on CD derivation
%         Z = (1 - 2*alpha*gam)*Beta(i) + 2*alpha*(X(:,i)'*Y*theta - X(:,i)'*X*Beta);
%         Beta(i) = sign(Z)*max(abs(Z) - alpha*lam, 0);
       
        
%         
        %Derivation from GLM paper
        Z = 2*X(:,i)'*(Y*(theta)) - 2*X(:,i)'*(X*Beta - Beta(i)*X(:,i));
        Beta(i) = (sign(Z).*max(abs(Z) - lam, 0))/(2*(gam + n-1));

    end
    
    Db = norm(Beta-b_old)/norm(Beta);
    %Should I change the convergence criteria? This did not work.
%     df = A*Beta - d;
%     
%     err = zeros;
%     % Initialize cardinality of support.
%     card = 0;
%    
%     % For each i, update error if i in the support.
%     for i=1:p
%         if abs(Beta(i)) > 1e-12    % i in supp(x).
%             % update cardinality.
%             card = card + 1; 
%             
%             % update error vector.
%             err(i) = -df(i) - lam*sign(Beta(i));
%         end
%     end
%     
%     if max(norm(df, inf) - lam, norm(err, inf)) < tol*p
%         % CONVERGED!!!
%         fprintf('Subproblem converged after %g iterations\n\n\n', betaIts);
%         break
%     end
    betaIts = betaIts + 1;
end
betaIts

end




%guess from using GLM paper
%         if j > 1 
%               Z = 2*alpha*(X(:,i)'*Y*theta - X(:,i)'*Y*Q(:,j-1));
%         else
%               Z = 2*alpha*(X(:,i)'*Y*theta - X(:,i)'*Y*Q(:,j));
%         end
%         Beta(i) = (sign(Z).*max(abs(Z) - alpha*lam, 0))/(1 + 2*gam);
%         
        %mix of GLM and derivation
%         Z = 2*alpha*(X(:,i)'*Y*theta - X(:,i)'*X*Beta);
%         Beta(i) = (sign(Z)*max(abs(Z) - alpha*lam, 0))/(1 + 2*gam);