function p = computeLogLikelihood(amplitudes,basis_function_projections)

p = sum(log(amplitudes*basis_function_projections./sum(amplitudes)));

end