function plotGaussianMixtureAndFreeEnergy(x,gaussian_parameters,amplitudes,...
    gaussian_param_est, amplitudes_est, temperature,boltzmann_const)


subplot(1,2,1)
hold on
gaussians = getGaussians(x,gaussian_parameters);

subplot(1,2,2)
hold on
gaussians2 = getGaussians(x,gaussian_param_est);
for i = 1:size(gaussians2,1)
    plot(x,amplitudes_est(i)*gaussians2(i,:),'k--');
end

plot(x, amplitudes_est*gaussians2, 'k','linewidth',2);
set(gca,'fontsize',14);
xlabel('Reaction coordinate');
ylabel('Probability density');
set(gca,'TickLabelInterpreter','latex');
title('b)');
set(gca,'ylim',[0,7])
set(gca,'xlim',[0,1])

subplot(1,2,1)
hold on
yyaxis left
gaussians = getGaussians(x,gaussian_parameters);
% for i = 1:size(gaussians,1)
%     plot(x,amplitudes(i)*gaussians(i,:),'k--');
% end

plot(x, amplitudes*gaussians, 'k','linewidth',2);
set(gca,'fontsize',14);
xlabel('Reaction coordinate');
ylabel('Probability density');
set(gca,'TickLabelInterpreter','latex');
title('a)');
set(gca,'ylim',[0,7])
set(gca,'xlim',[0,1])
set(gca,'Ycolor',[0,0,0]);

fe = -boltzmann_const*temperature*log(amplitudes*gaussians);
fe = fe - min(fe);
yyaxis right
hold on
plot(x, fe,'-','color',[0.6,0.6,0.6],'markerfacecolor','k','linewidth',2);
set(gca,'fontsize',13);
set(gca,'ylim',[0,4])
set(gca,'xlim',[0,1])
xlabel('Reaction coordinate');
ylabel('Free energy [kcal/mol]');
set(gca,'TickLabelInterpreter','latex');
set(gca,'Ycolor',[0.6,0.6,0.6]);
end
