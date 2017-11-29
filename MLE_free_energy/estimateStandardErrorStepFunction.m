function [standardError, free_energies] = estimateStandardErrorStepFunction(data, nMinBins, nMaxBins,...
    boltzmann_const, temperature, nTrajectories, evaluation_points)
% Computes the standard error of all (nTrajectories) trajectories using Bayesian inference.
% Returns the standard error as well as all energies at all data points.

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
free_energies = zeros(nTrajectories, nSamples);


% Prepare gaussians for energy calculation of complete trajectory
for i = 1:nTrajectories  
    if iscell(data)
        tmpTrajectory = data{i};
    else
        tmpTrajectory = allData((i-1)*nTestSamples + 1:i*nTestSamples);
    end
    
    % Estimate the best energy for the current trajectory
    free_energy = MLE_SF_FE(tmpTrajectory, ...
     temperature, boltzmann_const, nMinBins, nMaxBins, evaluation_points);
    
    free_energies(i,:) = free_energy';
    
end


standardError = sqrt(var(free_energies)'./nTrajectories);

% standard_error_fun = @(x) std(x)'./sqrt(nTrajectories);
% 
% 
% for j = 1:size(energies,2)
%     all_standard_errors = bootstrp(nBootstrapResamples, @(x)standard_error_fun(x), energies(:,j));
%     
%     % Compute standard error
%     standardError(j) = mean(all_standard_errors);
% end


end