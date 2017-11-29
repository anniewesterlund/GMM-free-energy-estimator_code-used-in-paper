function p = probability_density(ai,gaussian_parameters,xj)

mean_ = gaussian_parameters(:,1);
variance_ = gaussian_parameters(:,2).^2;
ai_sum = sum(ai);

gaussians = exp(-(mean_-xj).^2./(2.*variance_))./(sqrt(2.*pi.*variance_))./ai_sum;
p = ai*gaussians;

end