function [standardError, energies] = GM_grid_SE(data, nMinGaussians, nMaxGaussians,...
    boltzmann_const, temperature, nTrajectories, evaluation_points)
%
%   estimateStandardError.m
%
%   Computes the standard error of all (nTrajectories) trajectories using Bayesian inference.
%   Returns the standard error as well as all energies at all data points.
%   
%   Input:
%       - data: vector containing order parameter values obtained through
%       simulation.
%       - temperature: Temperature during simulation [K]
%       - boltzmann_const: boltzmann constant or gas constant [kcal/mol]
%       - nMinGaussians: The minimum number of Gaussian basis functions to
%       use.
%       - nMaxGaussians: The maximun number of Gaussian basis functions to
%       use.
%       - nTrajectories: number of trajectories.
%       - evaluation_points: points to evaluate the standard error at.
%
%   Output:
%       - standardError: vector with the estimated standard error.
%
%

if nargin == 5
    evaluation_points = [];
    nTrajectories = 3;
elseif nargin == 6
    evaluation_points = [];
end
if isempty(nTrajectories)
    nTrajectories = 3;
end


nSamples = 0;
if iscell(data)
    % Find min and max in all trajectories and set parameters 
    nTrajectories = length(data);
    allData = data{1};
    minX = min(data{1});
    maxX = max(data{1});
    nSamples = length(data{1});
    for i = 2:length(data)
        allData = [allData;data{i}];
       if minX > min(data{i})
           minX = min(data{i});
       end
       if maxX < max(data{i})
           maxX = max(data{i});
       end
       nSamples = nSamples + length(data{i});
    end
else
    % Find min and max in all trajectories
    allData = data;
    minX = min(data);
    maxX = max(data);
    nSamples = length(data);
end

if isempty(evaluation_points)
    evaluation_points = allData;
else
    nSamples = length(evaluation_points);    
end


nTestSamples = floor(nSamples/nTrajectories);
energies = zeros(nTrajectories, nSamples);

% Prepare gaussians for energy calculation of complete trajectory
for i = 1:nTrajectories  
    if iscell(data)
        tmpTrajectory = data{i};
    else
        tmpTrajectory = allData((i-1)*nTestSamples + 1:i*nTestSamples);
    end
        
    
    % Estimate the best energy for the current trajectory
    [~, ~, gaussian_parameters, amplitudes] = MLE_GM_grid_FE(tmpTrajectory,...
        temperature, boltzmann_const, nMinGaussians, nMaxGaussians);
    
    % Compute the energy profile for all points
    p = probability_density_all_gaussians(amplitudes,getGaussians(evaluation_points,...
        gaussian_parameters));
    p(p < 1e-5) = 1e-5;
    
    tmpEnergy = -temperature.*boltzmann_const.*log(p);
    
    energies(i,:) = tmpEnergy';
    
end

standardError = sqrt(var(energies)'./nTrajectories);

end