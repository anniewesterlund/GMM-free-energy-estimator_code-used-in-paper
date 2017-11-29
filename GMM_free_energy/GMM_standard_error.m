function standard_error = GMM_standard_error(data, nMinGaussians, ...
    nMaxGaussians, temperature, boltzmann_const, stopCriterion, nTrajectories, evaluation_points)

% Computes the standard error of all (nTrajectories) trajectories.
% Returns the standard error of free energies.

if nargin == 5
    stopCriterion = 1e-6;
    evaluation_points = [];
    nTrajectories = 3;
elseif nargin == 6
    nTrajectories = 3;
    evaluation_points = [];
elseif nargin == 7
    evaluation_points = [];
end

if isempty(stopCriterion)
    stopCriterion = 1e-6;
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


% Prepare for free energy calculation of complete trajectory
for i = 1:nTrajectories  
    if iscell(data)
        tmpTrajectory = data{i};
    else
        tmpTrajectory = allData((i-1)*nTestSamples + 1:i*nTestSamples,:);
    end
    
    % Estimate the best energy for the current trajectory
    [free_energy, ~, amplitudes, means, covariances] = GMM_free_energy_estimator(tmpTrajectory, nMinGaussians, ...
        nMaxGaussians, temperature, boltzmann_const, stopCriterion,evaluation_points);
    
    free_energies(i,:) = free_energy';
    
end

standard_error = sqrt(var(free_energies)'./nTrajectories)';

end