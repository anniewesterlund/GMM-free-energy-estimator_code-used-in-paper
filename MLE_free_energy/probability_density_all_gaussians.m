function p = probability_density_all_gaussians(ai,gaussians)


p = ai*gaussians./sum(ai);

end