function [free_energy, k_inferred, density, maxLogLikelihood] = MLE_kNN_FE(x,...
    temperature, boltzmann_const, nMinK, nMaxK)
%
%   MLE_KNN_FE.m
%
%   Estimates the 1D free energy along a reaction coordinate using Bayesian
%   inference and KNN density estimation. Infers the number of basis 
%   functions to use to prevent overfitting.
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
%
%   Output:
%       - free_energy: The estimated free energy
%       - k_inferred: The number of neighbors used to estimate the
%       profile.
%       - k_inferred: Used parameters for basis functions.
%
%       If no predictive inference is wanted, set nMinK and
%       nMaxK to the same value.
%


k_inferred = nMinK;
maxLogLikelihood = -inf;

half_size = floor(length(x)*0.5);


x_train_set = x(1:half_size);
x_validation_set = x(half_size+1:end);
if nMinK ~= nMaxK
    % Find optimal number of Gaussian basis functions
    for k = nMinK:nMaxK
        
        % Estimate the pdf
        density = kNNDensity(x_train_set, k);
        
        % Use model on validation set
        prob = logLikelihoodKNN(x_validation_set, x_train_set, density);

        % Update the number of k if the new model is more
        % likely based on the validation set.
        if prob > maxLogLikelihood
            maxLogLikelihood = prob;
            k_inferred = k;
        end
    end
end

% Estimate the pdf with all data
density = kNNDensity(x, k_inferred);
density(density < 1e-5) = 1e-5;

% Compute free energy
free_energy = -temperature.*boltzmann_const.*log(density);

end