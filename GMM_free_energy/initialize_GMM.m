function [amplitudes, means, covariances] = initialize_GMM(x, nGaussians)

nPoints = size(x,1);
nDims = size(x,2);

xMin = min(x);
xMax = max(x);

% Set means from points
randomInds = randperm(nPoints);
means = x(randomInds(1:nGaussians),:);

% Set amplitudes
amplitudes= 1/nGaussians.*ones(1,nGaussians);
 
tmpCov = zeros(nDims,nDims);
covariances = cell(nGaussians,1);

% Set initial covariance guess as the covariance of the entire dataset
tmpCov = cov(x);
for iG = 1:nGaussians
    covariances{iG} = tmpCov;
end

end