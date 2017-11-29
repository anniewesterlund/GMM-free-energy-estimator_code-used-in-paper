function [standardError, energies] = estimateStandardErrorKNN(data, nMinK, nMaxK,...
    boltzmann_const, temperature, nTrajectories)
% Computes the standard error of all (nTrajectories) trajectories using Bayesian inference.
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
       nSamples = nSamples + length(data{i});
    end
else
    % Find min and max in all trajectories
    allData = data;
    minX = min(data);
    maxX = max(data);
    nSamples = length(data);
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
    [~, ~, density] = MLE_kNN_FE(tmpTrajectory, ...
     temperature, boltzmann_const, nMinK, nMaxK);
    
    % Compute the energy profile for all points
    p = interp1(tmpTrajectory, density, allData);    
    p(p < 1e-5) = 1e-5;
    
    tmpEnergy = -temperature.*boltzmann_const.*log(p);
    
    energies(i,:) = tmpEnergy';
    
end

standardError = zeros(size(allData));

for i = 1:size(energies,2)
    tmpEnergy = energies(:,i);
    ind = ~isnan(tmpEnergy);
    nInd = sum(ind);
    standardError(i) = sqrt(var(tmpEnergy(ind))'./nInd);
end