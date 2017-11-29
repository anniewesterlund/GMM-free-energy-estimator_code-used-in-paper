function logLikelihood = logLikelihoodKNN(x_validation_set, x_train, density_train)

density_validation = interp1(x_train, density_train, x_validation_set);
density_validation(isnan(density_validation)) = 1e-5;
density_validation(density_validation < 1e-5) = 1e-5;

logLikelihood = sum(log(density_validation));

end