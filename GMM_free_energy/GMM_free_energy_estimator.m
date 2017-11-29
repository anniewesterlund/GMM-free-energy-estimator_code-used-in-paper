function [free_energy, logLikelihood, amplitudes, means, covariances] = ...
    GMM_free_energy_estimator(x, nMinGaussians, nMaxGaussians, temperature, boltzmann_const, stopCriterion,evaluation_points)
%
%   GMM_free_energy_estimator.m
%   Estimate free energy along reaction coordinate with Gaussian mixture
%   model and cross-validation
%   Input:
%       - x: order parameter values 
%       - nMinGaussians: minimum number of Gaussians in model
%       - nMaxGaussians: maximum number of Gaussians in model
%       - temperature
%       - boltzmann_cost
%       - stopCriterion: Stop criterion for loglikelihood difference in GMM
%       - evaluation_points: Points to evaluate the resulting free energy
%       model at
%
%   Output:
%       - free_energy
%       - logLikelihood: LLH of validation set
%       - ampitudes: amplitudes of Gaussian components
%       - means
%       - covariances


half_set = floor(length(x)*0.5);

x_training = x(1:half_set,:);
x_validation = x(half_set+1:end,:);

bestLogLikelihood = -inf;
nGaussiansFinal = nMinGaussians;

if nargin == 5
    stopCriterion = 1e-6;
    evaluation_points = x;
elseif nargin == 6
    if isempty(stopCriterion)
       stopCriterion = 1e-6; 
    end
    evaluation_points = x;
end

if nMinGaussians ~= nMaxGaussians
    for nGaussians = nMinGaussians:nMaxGaussians
        disp(['GMM nGaussians: ',num2str(nGaussians)])
        bestTrainLogLikelihood = -inf;
        
        [trainAmplitudes, trainMeans, trainCovariances] = GMM(x_training, nGaussians, stopCriterion);
        
        % Use validation set to score the model
        projections_val = GMM_get_Gaussians(x_validation, trainMeans, trainCovariances);
        logLikelihood = GMM_log_likelihood(trainAmplitudes,projections_val);
            
        if logLikelihood > bestLogLikelihood
            bestLogLikelihood = logLikelihood;
            nGaussiansFinal = nGaussians;
        end
    end
end

disp('GMM final optimization');
logLikelihood = bestLogLikelihood;
nGaussians = nGaussiansFinal;

[amplitudes, means, covariances] = GMM(x, nGaussians, stopCriterion);

if size(x,2) ~= size(evaluation_points,2) && size(x,2) == size(evaluation_points,1)
    evaluation_points = evaluation_points';
end

gaussians = GMM_get_Gaussians(evaluation_points, means, covariances);
density = amplitudes*gaussians;
density(density < 1e-5) = 1e-5;

free_energy = -temperature*boltzmann_const*log(density);
free_energy = free_energy - min(free_energy);

end