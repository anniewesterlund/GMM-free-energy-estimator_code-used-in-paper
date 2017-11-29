function [gaussian_means, gaussian_covariance, start_amplitudes] = setGaussianParameters2D(dataX, ...
    dataY, nGaussiansX, nGaussiansY)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Get the means, covariances and amplitudes of the initial GM
%   distribution for free energy estimation.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sigma_scale = 0.7;

gaussian_covariance = eye(2,2);
gaussian_means = zeros(nGaussiansX*nGaussiansY,2);

maxDataX = max(dataX);
minDataX = min(dataX);

maxDataY = max(dataY);
minDataY = min(dataY);

spacingsX = (maxDataX-minDataX)/nGaussiansX;
spacingsY = (maxDataY-minDataY)/nGaussiansY;

X_mean = minDataX+spacingsX/2:spacingsX:(minDataX + spacingsX*nGaussiansX);
Y_mean = minDataY+spacingsY/2:spacingsY:(minDataY + spacingsY*nGaussiansY);

sigma_X = sigma_scale*spacingsX;
sigma_Y = sigma_scale*spacingsY;

% Set Gaussian covariances
gaussian_covariance(1,1) = sigma_X.^2;
gaussian_covariance(2,2) = sigma_Y.^2;


% Set Gaussian mean values
for i = 1:nGaussiansX
    for j = 1:nGaussiansY
        gaussian_means((i-1)*nGaussiansY + j,:) = [X_mean(i),Y_mean(j)];
    end
end


% Start guess amplidtudes
start_amplitudes = ksdensity([dataX,dataY],gaussian_means,'width',[sigma_X,sigma_Y]);
start_amplitudes = start_amplitudes'./sum(start_amplitudes);

% start_amplitudes = 1/(nGaussians*nGaussians).*ones(1,nGaussians*nGaussians);


end