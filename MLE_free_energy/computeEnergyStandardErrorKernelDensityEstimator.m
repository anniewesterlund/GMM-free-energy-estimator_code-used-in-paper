function [standardError, energies] = computeEnergyStandardErrorKernelDensityEstimator(data,...
    boltzmann_const, temperature, nTrajectories, bandwidth, evaluationPoints)

% Computes the standard error of all (nTrajectories) trajectories.
% Returns the standard error as well as all energies at all data points.

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
       
    end
else
    % Find min and max in all trajectories
    allData = data;
    minX = min(data);
    maxX = max(data);
end

nSamples = length(evaluationPoints);
nTestSamples = floor(nSamples/nTrajectories);
energies = zeros(nTrajectories, nSamples);

% Set Gaussian parametes (mean + standard deviation + start guess on amplitudes)
for i = 1:nTrajectories  
    if iscell(data)
        tmpTrajectory = [data{i}; minX; maxX];
    else
        tmpTrajectory = [allData((i-1)*nTestSamples + 1:i*nTestSamples),minX,maxX];
    end
    
    % Kernel density estimation
    p = ksdensity(tmpTrajectory,evaluationPoints,'width',bandwidth);
    p(p < 1e-5) = 1e-5;
    
    % Compute free energy
    tmpEnergy = -temperature.*boltzmann_const.*log(p);
    energies(i,:) = tmpEnergy';
    
end

% Compute standard error
standardError = sqrt(var(energies)')./sqrt(nTrajectories);
end