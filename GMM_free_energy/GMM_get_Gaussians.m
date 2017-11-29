function gaussians = GMM_get_Gaussians(x, gaussian_means, gaussian_covariances)

nGaussians = size(gaussian_means,1);
nDims = size(x,2);
gaussians = zeros(nGaussians,size(x,1));

if nDims ~= 1
    for iG = 1:nGaussians
        gaussians(iG,:) = mvnpdf(x,gaussian_means(iG,:),gaussian_covariances{iG});
    end
else
    for iG = 1:nGaussians
        gaussians(iG,:) = mvnpdf(x,gaussian_means(iG,:),[gaussian_covariances{iG}]);
    end
end

end

