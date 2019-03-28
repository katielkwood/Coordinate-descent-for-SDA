function [Beta, betaIts ] = betaCoordDesc(b0, j, alpha, X, Y, theta, Q, Om, lam, gam, maxits, tol, d, A)


[n, p] = size(X);
Db = 1;
b_old = b;
Beta = b;
betaIts = 0;
while (Db > tol && betaIts < maxits)
    b_old = Beta;
    
    for j = 1:length(Beta)
               
        %based on CD derivation
%         Z = (1 - 2*alpha*gam)*Beta(j) + 2*alpha*(X(:,j)'*Y*theta - X(:,j)'*X*Beta);
%         Beta(j) = sign(Z)*max(abs(Z) - alpha*lam, 0);
       
        
%         
        %Derivation from GLM paper
        Z = 2*X(:,j)'*(Y*(theta)) - 2*X(:,j)'*(X*Beta - Beta(j)*X(:,j));
        Beta(j) = (sign(Z).*max(abs(Z) - lam, 0))/(2*(gam + n-1));

    end
   
    
    Db = norm(Beta-b_old)/norm(Beta);
    
    betaIts = betaIts + 1;
end


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
%         Beta(i) = (sign(Z)*max(abs(Z) - alpha*lam, 0))/(1 + 2*gam); 0))/(1 + 2*gam);