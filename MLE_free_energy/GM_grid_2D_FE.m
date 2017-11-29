function [X, Y, free_energy, nOptimalGaussiansX, nOptimalGaussiansY, logLikelihood, amplitudes, gaussians] = GM_grid_2D_FE(x1, x2, temperature, ...
    boltzmann_const, nMinGaussians, nMaxGaussians, nGrids, x_limits, y_limits)
%
%   estimateBayesianEnergyLandscape2D.m
%
%   Estimates the 2D free energy along a reaction coordinate using Bayesian
%   inference. Optimizes the number of basis functions to use to prevent
%   overfitting.
%   
%   Input:
%       - x1: [nx1] vector containing order parameter values obtained through
%       simulation.
%       - x2: [nx1] vector containing order parameter values obtained through
%       simulation.
%       - temperature: Temperature during simulation [K]
%       - boltzmann_const: boltzmann constant or gas constant [kcal/mol]
%       - nMinGaussians: The minimum number of Gaussian basis functions to
%       use.
%       - nMaxGaussians: The maximun number of Gaussian basis functions to
%       use.
%       - nGrids: Number of grids for the surface where the FE is
%       estimated.
%       - x_limits: vector with minX and maxX for final estimated surface.
%       - y_limits: vector with minY and maxY for final estimated surface.
%
%
%   Output:
%       - X: [nGrids x nGrids] x-coordinates
%       - Y: [nGrids x nGrids] y-coordinates
%       - free_energy: [nGrids x nGrids] The estimated free energy at X,Y
%       coordinates.
%       - nOptimalGausisansX: The number of Gaussians used along x-axis.
%       - nOptimalGausisansY: The number of Gaussians used along y-axis.
%       - amplitudes: Basis function amplitudes found through optimization.
%       - gaussians: The value of the gaussians.
%
%       If no predictive inference is wanted, set nMinGaussians and
%       nMaxGaussians to the same value.
%   
%

nOptimalGaussiansX = nMinGaussians;
nOptimalGaussiansY = nMinGaussians;

maxProbability = -inf;

half_size = floor(length(x1)/2);        

% Prepare Gaussians
tmpOP = [x1,x2];

x_test_set = tmpOP(1:half_size,:);
x_validation_set = tmpOP(half_size+1:end,:);

if nMinGaussians ~= nMaxGaussians
    % Find optimal number of basis functions
    for nGaussiansX = nMinGaussians:nMaxGaussians
        for nGaussiansY = nGaussiansX
            disp([nGaussiansX,nGaussiansY])
            % Set Gaussian parametes (mean + standard deviation + start guess on amplitudes)
            [gaussian_means, gaussian_covariance, start_amplitudes] = setGaussianParameters2D(x_test_set(:,1), ...
                x_test_set(:,2), nGaussiansX, nGaussiansY);
            
            gaussians = getGaussians2D(x_test_set, gaussian_means, gaussian_covariance);
            
            disp('Start optimization');
            amplitudes = estimateProbabilityDensity(start_amplitudes, gaussians);
            
            % Use model on validation set
            validation_gaussians = getGaussians2D(x_validation_set, gaussian_means, gaussian_covariance);
            prob = computeLogLikelihood(amplitudes,validation_gaussians);
            
            % If validation set likelihood is larger than before, update
            % the number of basis functions.
            if prob > maxProbability
                maxProbability = prob;
                nOptimalGaussiansX = nGaussiansX;
                nOptimalGaussiansY = nGaussiansY;
            end
            
        end
    end
    disp('Start final optimization');
end

logLikelihood = maxProbability;

% Estimate pdf
[gaussian_means, gaussian_covariance, start_amplitudes] = setGaussianParameters2D(tmpOP(:,1), ...
    tmpOP(:,2), nOptimalGaussiansX, nOptimalGaussiansY);

gaussians = getGaussians2D(tmpOP, gaussian_means, gaussian_covariance);

amplitudes = estimateProbabilityDensity(start_amplitudes, gaussians);

minX = x_limits(1);
maxX = x_limits(end);
minY = y_limits(1);
maxY = y_limits(end);

dx = (maxX-minX)/nGrids;
dy = (maxY-minY)/nGrids;

xVector = minX:dx:maxX;
yVector = minY:dy:maxY;

[X, Y] = meshgrid(xVector,yVector);

% Prepare Gaussians
tmpOP = [X(:),Y(:)];

gaussians = getGaussians2D(tmpOP, gaussian_means, gaussian_covariance);

% Compute resulting free energies
tmp_pdf = probability_density_all_gaussians(amplitudes,gaussians);
tmp_pdf(tmp_pdf < 1e-5) = 1e-5;
tmp_pdf = reshape(tmp_pdf,length(yVector),length(xVector));

free_energy= -temperature.*boltzmann_const.*log(tmp_pdf);

% Normalize
free_energy = free_energy - min(free_energy(:));

end