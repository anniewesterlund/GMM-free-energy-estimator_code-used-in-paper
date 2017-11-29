function standardError = GaussianProjectionStandardError(data,...
    temperature, boltzmann_const, nGaussians)


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
        
    [~, ~, gaussian_parameters, ...
         amplitudes] = GaussianProjectionFreeEnergy(tmpTrajectory,...
        temperature, boltzmann_const, nGaussians);
    
    % Compute weighted FE GM evaluated at all points
    density = amplitudes*getGaussians(allData,gaussian_parameters);
    
    density(density < 1e-5) = 1e-5;
    tmpValue = -temperature*boltzmann_const*log(density);
    tmpEnergy = tmpValue - min(tmpValue);
    
    energies(i,:) = tmpEnergy';
    
end

standardError = sqrt(var(energies)'./nTrajectories);

end