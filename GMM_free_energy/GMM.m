function [amplitudes, means, covariances, logLikelihood] = GMM(x, nGaussians, stopCriterion)
% Gaussian mixture model for estimating amplitudes, means and covariances
% given the data in x.

if nargin == 2
    stopCriterion = 1e-4;
end

% Initialize model
[amplitudes, means, covariances] = initialize_GMM(x, nGaussians);

% Compute projections and loglikelihood
projections = GMM_get_Gaussians(x, means, covariances);
previousLogLikelihood = GMM_log_likelihood(amplitudes, projections);

logLikelihood = -inf;
counter = 1;

while abs(previousLogLikelihood-logLikelihood) > stopCriterion
    % Set labels according to parameters
    membership_weights = GMM_expectation(amplitudes, projections);
    
    previousLogLikelihood = logLikelihood;
    
    % Maximization step
    [means, covariances, amplitudes, logLikelihood, projections] = GMM_maximization(x,...
        means, covariances, membership_weights);
    counter = counter + 1;
end

end