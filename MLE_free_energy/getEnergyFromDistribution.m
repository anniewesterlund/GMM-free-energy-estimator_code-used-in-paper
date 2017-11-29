function free_energy = getEnergyFromDistribution(amplitudes, gaussian_parameters, ...
    x, temperature, boltzmann_const)

gaussians = zeros(size(gaussian_parameters,1),length(x));
variance_ = gaussian_parameters(:,2).^2;
mean_ = gaussian_parameters(:,1);

% Prepare gaussians
for iG = 1:size(gaussian_parameters,1)
    gaussians(iG,:) = exp(-(mean_(iG)-x').^2./(2.*variance_(iG)))./(sqrt(2.*pi.*variance_(iG)));
end

free_energy = -temperature.*boltzmann_const.*log(probability_density_all_gaussians(amplitudes,gaussians));

end