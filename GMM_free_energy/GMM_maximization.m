function [means, covariances, amplitudes, logLikelihood, projections] = GMM_maximization(x, means, covariances, ...
    membershipWeights)

% Maximization step of GMM EM algorithm
nPoints = size(x,1);
nDims = size(x,2);
nGaussians = length(covariances);

% Update the amplitudes
componentWeights = sum(membershipWeights,2);
amplitudes = componentWeights'/nPoints;
amplitudes = amplitudes/sum(amplitudes);


for iG = 1:nGaussians
    
    % Update the means
    weightedPoints = membershipWeights(iG,:)*x;
    
    means(iG,:) = 1./componentWeights(iG)*weightedPoints;
    
    % Update covariances
    tmpExp = bsxfun(@minus,x,means(iG,:))';
    tmpExp2 = bsxfun(@times,tmpExp,membershipWeights(iG,:));
    tmpCov = tmpExp*tmpExp2';
    
    covariances{iG} = 1./componentWeights(iG)*tmpCov + eye(nDims)*1e-7;
    
end

projections = GMM_get_Gaussians(x, means, covariances);
logLikelihood = GMM_log_likelihood(amplitudes, projections);

end

