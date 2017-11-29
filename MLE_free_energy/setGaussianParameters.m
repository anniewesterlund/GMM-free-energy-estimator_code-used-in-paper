function [gaussian_parameters, start_amplitudes] = setGaussianParameters(data, nGaussians)

% Golden scaling factor of Gaussians
sigma_scale = 0.71;

spacings = (max(data)-min(data))/nGaussians;

% Gaussian means
gaussian_parameters(:,1) = min(data)+spacings/2:spacings:(min(data) + spacings*nGaussians);
% Gaussian standard deviations
gaussian_parameters(:,2) = sigma_scale*spacings;

% Estimate density with KDE to get a first good approximation of the
% amplitudes
pTmp=ksdensity(data,gaussian_parameters(:,1),...
    'width',sigma_scale*spacings,'function','pdf');

start_amplitudes = pTmp';

end