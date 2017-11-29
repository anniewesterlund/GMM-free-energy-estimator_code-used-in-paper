function [free_energy, nOptimalGaussians, gaussian_parameters, amplitudes, maxLogLikelihood] = MLE_GM_grid_FE(x,...
    temperature, boltzmann_const, nMinGaussians, nMaxGaussians, evaluation_points)
%
%   MLE_GM_grid_FE.m
%
%   Estimates the 1D free energy along a reaction coordinate using Bayesian
%   inference. Optimizes the number of basis functions to use to prevent
%   overfitting.
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
%       - evaluation_points: points to evaluate free energy at
%
%   Output:
%       - free_energy: The estimated free energy
%       - nOptimalGausisans: The number of Gaussians used to estimate the
%       profile.
%       - gaussian_parameters: Used parameters for basis functions.
%       - amplitudes: Basis function amplitudes found through optimization.
%
%

if nargin == 4
    nMaxGaussians = nMinGaussians;
    evaluation_points = [];
elseif nargin == 5
    evaluation_points = [];
end

if isempty(nMaxGaussians)
    nMaxGaussians = nMinGaussians;
end

nOptimalGaussians = nMinGaussians;
maxLogLikelihood = -inf;
sigma_scale_vec = linspace(0.5,0.9,7);
bestSigmaScale = 0.5;
bestShift = 0;

% half_size = floor(length(x)/2);
half_size = floor(length(x)*0.5);

x_train_set = x(1:half_size);
x_validation_set = x(half_size+1:end);
if nMinGaussians ~= nMaxGaussians
    % Find optimal number of Gaussian basis functions
    for nGaussians = nMinGaussians:nMaxGaussians
        disp(['nGaussians: ',num2str(nGaussians)]);
        [gaussian_parameters_orig, start_amplitudes] = setGaussianParameters(x_train_set, nGaussians);
        
        dx = (gaussian_parameters_orig(2,1)-gaussian_parameters_orig(1,1));
        shift_vector = 0:dx/(4):dx;
        gaussian_parameters = gaussian_parameters_orig;
        
        for shift = shift_vector
            gaussian_parameters(:,1) = gaussian_parameters_orig(:,1) + shift;
            
            for sigma_scale = sigma_scale_vec
                gaussian_parameters(:,2) = dx*sigma_scale;
                
                % Estimate the pdf
                train_gaussians = getGaussians(x_train_set,gaussian_parameters);
                
                amplitudes = estimateProbabilityDensity(start_amplitudes, train_gaussians);
                
                % Use model on validation set
                validation_gaussians = getGaussians(x_validation_set, gaussian_parameters);
                prob = computeLogLikelihood(amplitudes,validation_gaussians);
                
                % Update the optimal number of Gaussians if the new model is more
                % likely based on the validation set.
                if prob > maxLogLikelihood
                    maxLogLikelihood = prob;
                    nOptimalGaussians = nGaussians;
                    bestShift = shift;
                    bestSigmaScale = sigma_scale;
                end
            end
        end
    end
end

% Estimate the pdf with all data
[gaussian_parameters, start_amplitudes] = setGaussianParameters(x, nOptimalGaussians);
gaussian_parameters(:,1) = gaussian_parameters(:,1) + bestShift;
gaussian_parameters(:,2) = bestSigmaScale*(gaussian_parameters(2,1)-gaussian_parameters(1,1));

gaussians = getGaussians(x,gaussian_parameters);
amplitudes = estimateProbabilityDensity(start_amplitudes, gaussians);

if ~isempty(evaluation_points)
    gaussians = getGaussians(evaluation_points, gaussian_parameters);
end

p = probability_density_all_gaussians(amplitudes,gaussians);
p(p < 1e-5) = 1e-5;

% Compute free energy
free_energy = -temperature.*boltzmann_const.*log(p);

% Shift free energy
free_energy = free_energy - min(free_energy);

end