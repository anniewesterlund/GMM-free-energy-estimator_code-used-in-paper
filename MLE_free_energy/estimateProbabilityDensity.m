function amplitudes = estimateProbabilityDensity(start_amplitudes, basis_function_projections, stopCriterion)
%
%   estimateProbabilityDensity.m
%
%   Estimates the probability density given initial guess and value of
%   each point at the basis funcitons.
%   
%   Input:
%       - start_amplitudes: initial guess of amplitudes
%       - basis_functions: value of each order parameter point at each
%       basis function.
%
%   Output:
%       - amplitudes: optimized amplitudes.
%
amplitudes = start_amplitudes/sum(start_amplitudes);

if nargin == 2
    stopCriterion = 1e-6;
end

% Compute projections and loglikelihood
previousLogLikelihood = computeLogLikelihood(amplitudes, basis_function_projections);

logLikelihood = -inf;
nPoints = size(basis_function_projections,2);
nBasisFunctions = size(basis_function_projections,1);
    
while abs(previousLogLikelihood-logLikelihood) > stopCriterion
    
    % Expectation step
    membership_weights = ones(nBasisFunctions,nPoints);
    normalizationFactor = amplitudes*basis_function_projections;
    
    for iG = 1:nBasisFunctions
        membership_weights(iG,:) = amplitudes(iG)*basis_function_projections(iG,:)./normalizationFactor;
    end
        
    previousLogLikelihood = logLikelihood;
    
    % Maximization step
    componentWeights = sum(membership_weights,2);
    amplitudes = componentWeights'/nPoints;
    amplitudes = amplitudes/sum(amplitudes);
    
    % Log-likelihood for current parameter set
    logLikelihood = computeLogLikelihood(amplitudes, basis_function_projections);
    
end

end