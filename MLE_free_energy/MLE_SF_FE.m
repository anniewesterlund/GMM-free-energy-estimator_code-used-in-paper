function [free_energy, nOptimalBins, amplitudes, edges, maxLogLikelihood] = MLE_SF_FE(x, ...
    temperature, boltzmann_const, nMinBins, nMaxBins, evaluation_points)

if nargin == 5
    evaluation_points = [];
end

nOptimalBins = nMinBins;
maxLogLikelihood = -inf;

half_size = floor(length(x)*0.5);

x_train_set = x(1:half_size);
x_validation_set = x(half_size+1:end);

if nMinBins ~= nMaxBins
    for nBins = nMinBins:nMaxBins

        % Estimate the pdf
        [start_amplitudes, edges_orig] = hist(x_train_set,nBins);
        start_amplitudes = start_amplitudes./length(x_train_set);
        
        dx = (edges_orig(2)-edges_orig(1));
        shift_vector = 0:dx/(4):dx;
        
        for shift = shift_vector
            edges = edges_orig + shift;

            projections = projectDataOnStepFunction(x_train_set, edges);
            amplitudes = estimateProbabilityDensity(start_amplitudes, projections);

            % Use model on validation set    
            projections_val = projectDataOnStepFunction(x_validation_set, edges);

            logLikelihood = computeLogLikelihood(amplitudes,projections_val);

            if logLikelihood > maxLogLikelihood
                maxLogLikelihood = logLikelihood;
                nOptimalBins = nBins;
            end
        end
    end
end

% Estimate the pdf with all data
[start_amplitudes, edges] = hist(x, nOptimalBins);
start_amplitudes(start_amplitudes==0) = 1e-5;
start_amplitudes = start_amplitudes./length(x);

projections = projectDataOnStepFunction(x, edges);
tmp = start_amplitudes*projections;

amplitudes = estimateProbabilityDensity(start_amplitudes, projections);

if ~isempty(evaluation_points)
    projections = projectDataOnStepFunction(evaluation_points, edges);
end

density = amplitudes*projections;
density(isnan(density)) = 1e-5;
density(density < 1e-5) = 1e-5;

free_energy = -temperature.*boltzmann_const.*log(density);

free_energy = free_energy - min(free_energy);

end