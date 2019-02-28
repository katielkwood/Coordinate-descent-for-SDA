function [Beta] = betaCoordDesc(b0, j, alpha, X, Y, theta, Q, Om, lam, gam, maxits, tol)

Db = 1;
b_old = b0;
Beta = b0;
betaIts = 0;
while (Db > tol && betaIts < maxits)
    
    b_old = Beta;
    
    for i = 1:length(Beta)
               
        %based on CD derivation
        %Z = (1 - 2*alpha*gam)*Beta(i) + 2*alpha*(X(:,i)'*Y*theta - X(:,i)'*X*Beta);
        %Beta(i) = sign(Z)*max(abs(Z) - alpha*lam, 0);
       
        %guess from using GLM paper
        if j > 1 
              Z = 2*alpha*(X(:,i)'*Y*theta - X(:,i)'*Y*Q(:,j-1));
        else
              Z = 2*alpha*(X(:,i)'*Y*theta - X(:,i)'*Y*Q(:,j));
        end
        Beta(i) = (sign(Z).*max(abs(Z) - alpha*lam, 0))/(1 + 2*gam);
%         
        %mix of GLM and derivation
        %Z = 2*alpha*(X(:,i)'*Y*theta - X(:,i)'*X*Beta);
        %Beta(i) = (sign(Z)*max(abs(Z) - alpha*lam, 0))/(1 + 2*gam);
    end
    
    Db = norm(Beta-b_old)/norm(Beta);
    betaIts = betaIts + 1;
end
betaIts

end