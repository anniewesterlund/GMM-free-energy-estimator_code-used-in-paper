function plotGaussianMixture(x,gaussian_parameters,amplitudes)

hold on
gaussians = getGaussians(x,gaussian_parameters);
for i = 1:size(gaussians,1)
    plot(x,amplitudes(i)*gaussians(i,:),'k','linewidth',2);
end
plot(x, amplitudes*gaussians, 'k--');
set(gca,'fontsize',13);
xlabel('Collective variable');
ylabel('Density');
set(gca,'TickLabelInterpreter','latex');
end