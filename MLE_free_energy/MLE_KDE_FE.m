function [free_energy_max_likelihood, bestSigma, ...
    maxLogLikelihood, bestDensity] = MLE_KDE_FE(x,...
    temperature, boltzmann_const, sigmaMin, sigmaMax, evalPoints)
%
%   estimateBayesianKDE_FE.m
%
%   Estimates the 1D free energy along a reaction coordinate using Bayesian
%   inference to estimate bandwidth of KDE.
%
%   Input:
%       - x: vector containing order parameter values obtained through
%       simulation.
%       - temperature: Temperature during simulation [K]
%       - boltzmann_const: boltzmann constant or gas constant [kcal/mol]
%       - nMinGaussians: The minimum number of Gaussian basis functions to
%       use.
%       - nMaxGaussians: The maximun number of Gaussian basis functions to
%       use.
%       - evalPoints: points to evaluate the free energy at
%
%   Output:
%       - free_energy: The estimated free energy
%       - nOptimalGausisans: The number of Gaussians used to estimate the
%       profile.
%       - gaussian_parameters: Used parameters for basis functions.
%       - amplitudes: Basis function amplitudes found through optimization.
%
%       If no cross validation is wanted: set nMinGaussians and
%       nMaxGaussians to the same value.
%

distances = pdist2(x,x);
distances_eval = pdist2(evalPoints,x);

bestSigma = sigmaMin;
bestDensity = [];
maxLogLikelihood = -inf;
sigmaVec = sigmaMin:(sigmaMax-sigmaMin)/15:sigmaMax;

nIterations = length(sigmaVec);

% half_size = floor(length(x)/2);
half_size = floor(length(x)*0.5);

dist_train_set = distances(1:half_size,1:half_size);
dist_validation_set = distances(half_size+1:end,1:half_size);


nPoints = size(distances_eval,2);
nPointsTrain = half_size;

counter = 1;

% Find optimal number of Gaussian basis functions
for sigma = sigmaVec
    disp(['Sigma: ',num2str(sigma)]);
    
    gaussians = 1/(sqrt(2*pi)*sigma).*exp(-distances_eval.^2./(2.*sigma.^2));
    density = sum(gaussians,2)/nPoints;
    
    % Use model on validation set
    validation_gaussians = 1/(sqrt(2*pi)*sigma).*exp(-dist_validation_set.^2./(2.*sigma.^2));
    
    % Get probability density of validation set
    val_density = sum(validation_gaussians,2)/nPointsTrain;
    val_density(val_density < 1e-5) = 1e-5;
    logLikelihoods(counter) = sum(log(val_density));
        
    % Store density
    densities{counter} = density;
    
    density(density < 1e-5) = 1e-5;
    
    FE_tmp = -temperature.*boltzmann_const.*log(density);
    FE_tmp = FE_tmp - min(FE_tmp);
        
    % Update the maximum likelihood parameters, number of Gaussians
    % if the new model is more likely based on the validation set.
    if logLikelihoods(counter) > maxLogLikelihood
        maxLogLikelihood = logLikelihoods(counter);
        bestSigma = sigma;
        free_energy_max_likelihood = FE_tmp;
        bestDensity = density;
    end
    
    counter = counter + 1;
end


end