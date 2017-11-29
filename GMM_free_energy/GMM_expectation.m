function membershipWeights = GMM_expectation(amplitudes, projections)
% Expectation step of GMM
nPoints = size(projections,2);
nGaussians = size(projections,1);
membershipWeights = ones(nGaussians,nPoints);
normalizationFactor = amplitudes*projections;


for iG = 1:nGaussians
    membershipWeights(iG,:) = amplitudes(iG)*projections(iG,:)./normalizationFactor;
end


end