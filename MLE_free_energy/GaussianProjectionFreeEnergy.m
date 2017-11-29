function [free_energy_max_likelihood, nOptimalGaussians, max_likelihood_gaussian_parameters, ...
    max_likelihood_amplitudes] = GaussianProjectionFreeEnergy(x,...
    temperature, boltzmann_const, nGaussians)
%
%   Compute free energy by projecting onto Gaussian basis functions
%

half_size = floor(length(x)/2);

x_train_set = x(1:half_size);
x_validation_set = x(half_size+1:end);


% Training set parameters
gaussian_parameters_orig = setGaussianParameters(x_train_set, nGaussians);

% All data parameters
gaussian_parameters_all_orig = setGaussianParameters(x, nGaussians);

% % Estimate training set pdf
% train_gaussians = getGaussians(x_train_set,gaussian_parameters_orig);
% 
% amplitudes = sum(train_gaussians,2)';
% amplitudes = amplitudes/sum(amplitudes);
% 
% 
% % Use model on validation set
% validation_gaussians = getGaussians(x_validation_set, gaussian_parameters_orig);
% maxLogLikelihood = computeModelProbability(amplitudes,validation_gaussians);
% 
% % Estimate the pdf with all data
gaussians_all = getGaussians(x,gaussian_parameters_all_orig);
amplitudes = sum(gaussians_all,2)';

amplitudes = amplitudes/sum(amplitudes);

% amplitudes = hist(x, gaussian_parameters_all_orig(:,1));
% amplitudes = amplitudes/sum(amplitudes);


p = probability_density_all_gaussians(amplitudes,gaussians_all);
p(p < 1e-5) = 1e-5;

FE_tmp = -temperature.*boltzmann_const.*log(p);
FE_tmp = FE_tmp - min(FE_tmp);

nOptimalGaussians = nGaussians;
free_energy_max_likelihood = FE_tmp;
max_likelihood_gaussian_parameters = gaussian_parameters_all_orig;
max_likelihood_amplitudes = amplitudes;

end