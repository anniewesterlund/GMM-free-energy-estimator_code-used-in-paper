function gaussians = getGaussians2D(x, gaussian_means, gaussian_covariance)

nGaussians = size(gaussian_means,1);
gaussians = zeros(nGaussians,size(x,1));

for iG = 1:nGaussians
    gaussians(iG,:) = mvnpdf(x,gaussian_means(iG,:),gaussian_covariance);
end


end

%         tmpExponent = zeros(nGaussians,1);
%             gaussians(iG,:) = gaussians(iG,:)./sum(gaussians(iG,:));
% Same thing, but slower:
%         tmpX = bsxfun(@minus,gaussian_means(iG,:),tmpOP);
%         disp('gaussians:');
%         for i = 1:size(tmpOP,1)
% %             tmpExponent(i) = tmpX(i,:)*gaussian_covariance*tmpX(i,:)';
%             gaussians(iG,i) = exp(-1/2.*tmpX(i,:)*gaussian_covariance*tmpX(i,:)')./...
%             (2.*pi.*sqrt(gaussian_covariance(1,1)*gaussian_covariance(2,2)));
%             disp(gaussians(iG,i));
%         end
