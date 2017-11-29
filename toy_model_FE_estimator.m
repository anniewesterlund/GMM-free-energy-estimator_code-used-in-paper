clc
clear all
close all

warning off

% Set directories and file names
addpath('MLE_free_energy/');
addpath('GMM_free_energy/');

set(0,'defaulttextinterpreter','latex');

% initialize metric vectors
p_error_SF_all = [];
p_error_GM_all = [];
p_error_KDE_all = [];
p_error_kNN_all = [];
p_error_GMM_all = [];

error_SF_all = [];
error_GM_all = [];
error_KDE_all = [];
error_kNN_all = [];
error_GMM_all = [];

SE_SF_all = [];
SE_GM_all = [];
SE_KDE_all = [];
SE_kNN_all = [];
SE_GMM_all = [];

logli_SF_all = [];
logli_GM_all = [];
logli_KDE_all = [];
logli_kNN_all = [];
logli_GMM_all = [];

% Set parameters
nRuns = 6;

nMinGaussians = 12;
nMaxGaussians = 30;
nMinGaussiansGMM = 3;
nMaxGaussiansGMM = 18;
nMinBins = 12;
nMaxBins = 26;
minSigma = 0.003; 
maxSigma = 0.08;

nHistogramBins = 20;

nMinK = 3;
nMaxK = 20;

nDataSamples = 200;
nTrajectories = 10;
sampleDx = 1;

boltzmann_const = 1.9872041e-3;
temperature = 300;

kNN_color = [249,152,0]/255;
GMM_color = [180,120,250]/255;
trueColor = [0.6,0.6,0.6];
KDE_color = [0,0.5,1];
GM_color = [1,0,0];
SF_color = [0,0,0];

% Create the model density and landscape
mu = ([0.05; 0.12; 0.4; 0.5; 0.45; 0.47; 0.65; 0.74]/1.9 + 0.4);
sigma =  [0.2; 0.1; 0.013; 0.007; 0.007; 0.2; 0.03; 0.05]/1.9;

toy_model_amplitudes = [0.1,0.2,0.01,0.002,0.01,0.3,0.2,0.5];
toy_model_amplitudes = toy_model_amplitudes./sum(toy_model_amplitudes);

gaussian_parameters = [mu, sigma];
xVec = (0:0.01:1);
gaussians = getGaussians(xVec, gaussian_parameters);

minX = min(xVec);
maxX = max(xVec);

SIG = zeros(1,1,length(sigma));
for i = 1:length(sigma)
    SIG(1,1,i) = sigma(i)^2;
end

obj = gmdistribution(mu,SIG,toy_model_amplitudes);

% Compute the true free energy
trueDensity = toy_model_amplitudes*gaussians;
trueFE = -boltzmann_const*temperature*log(trueDensity);

for iRun = 1:nRuns
    
    % Sample trajectories
    trajectories = {};
    data = [];
    for i = 1:nTrajectories
        trajectories{i} = random(obj,nDataSamples);
        data = [data;trajectories{i}];
    end    
    
    % Estimate GM free energy
    if nMinGaussians == nMaxGaussians
        [MLE_FE_GM, nOptimalGaussians, gaussian_parameters_GM, ...
            amplitudes_GM_grid] = GaussianProjectionFreeEnergy(data,...
            temperature, boltzmann_const, nMinGaussians);
        
        % Estimate histogram free energy and standard error
        [SF_free_energy, SF_standard_error, ~, edgesHist, allEnergiesHist] =...
            histogram_FE(trajectories, boltzmann_const, temperature, nBins, 0);
        
    else
        [MLE_FE_GM,nOptimalGaussians, gaussian_parameters_GM, amplitudes_GM_grid, ...
            logLikelihood_MLE_GM] = MLE_GM_grid_FE(data,...
            temperature, boltzmann_const, nMinGaussians, nMaxGaussians);
    end
    
    if iRun == 1
        figure(10);
        plotGaussianMixtureAndFreeEnergy(xVec,gaussian_parameters,toy_model_amplitudes,...
            gaussian_parameters_GM, amplitudes_GM_grid, temperature,boltzmann_const);
    end
    
    
    [dataSort,indSort] = sort(data);
    
    % Compute standard error
    if nMinGaussians == nMaxGaussians
        standardErrorGM2 = GaussianProjectionStandardError(trajectories,...
            temperature, boltzmann_const, nMinGaussians);
        
        [GP_KDE, ~] = setGaussianParameters(data, nMinGaussians);
        
        kernelBandwidth = GP_KDE(1,2);
        
        % KDE on MD data
        [f_KDE, x_KDE_MD] = ksdensity(data,dataSort,'width',kernelBandwidth,'function','pdf');
        f_KDE(f_KDE < 1e-5) = 1e-5;
        kDensityFE = -temperature.*boltzmann_const.*log(f_KDE) - min(-temperature.*boltzmann_const.*log(f_KDE));
        
        [stdErrorKDE, allKDEEnergies] = computeEnergyStandardErrorKernelDensityEstimator(trajectories,...
            boltzmann_const, temperature, length(trajectories), kernelBandwidth, x_KDE_MD);
        
    else
        [standardErrorGM2, ~] = GM_grid_SE(trajectories, nMinGaussians, nMaxGaussians,...
            boltzmann_const, temperature, length(trajectories));
    end
    
    % Estimate kNN free energy
    [FE_kNN, k_inferred, kNN_density, logLikelihood_kNN] = MLE_kNN_FE(data,...
        temperature, boltzmann_const, nMinK, nMaxK);
    FE_kNN = FE_kNN(indSort);
    kNN_density = kNN_density(indSort);
    
    [SE_kNN, energies] = estimateStandardErrorKNN(trajectories, nMinK, nMaxK,...
        boltzmann_const, temperature, nTrajectories);
    SE_kNN = SE_kNN(indSort);
    
    % Estimate GMM free energy
    [GMM_free_energy, GMM_logLikelihood, GMM_amplitudes, GMM_means, GMM_covariances] = ...
        GMM_free_energy_estimator(data, nMinGaussiansGMM, nMaxGaussiansGMM, ...
        temperature, boltzmann_const, 1e-6);
    
    GMM_SE = GMM_standard_error(trajectories,  nMinGaussiansGMM, nMaxGaussiansGMM,...
        temperature, boltzmann_const, 1e-6);
    
    y_true = trueFE-min(trueFE);
    
    % Compute error to true profile
    % KDE on MD data
    if nMinGaussians == nMaxGaussians
        pKDE_tmp = ksdensity(data,xVec,'width',kernelBandwidth,'function','pdf');
        p_error_KDE = sqrt((trueDensity-pKDE_tmp).^2);
        pKDE_tmp(pKDE_tmp < 1e-5) = 1e-5;
        y_val_KDE = -temperature.*boltzmann_const.*log(pKDE_tmp) - min(-temperature.*boltzmann_const.*log(pKDE_tmp));
        error_KDE = sqrt((y_val_KDE-y_true).^2);
        
        y_val_hist = interp1(edgesHist,SF_free_energy,xVec);
        y_val_hist(isnan(y_val_hist)) = max(y_val_hist(~isnan(y_val_hist)));        
        error_hist = sqrt((y_val_hist-y_true).^2);       
    end
    
    p = amplitudes_GM_grid*getGaussians(xVec,gaussian_parameters_GM);
    p_error_GM = sqrt((trueDensity-p).^2);
    p(p < 1e-5) = 1e-5;
    tmpValue = -temperature*boltzmann_const*log(p);
    y_val_GM = tmpValue - min(tmpValue);
    
    p_GMM = GMM_amplitudes*GMM_get_Gaussians(xVec',GMM_means,GMM_covariances);
    p_error_GMM = sqrt((trueDensity-p_GMM).^2);
    p_GMM(p_GMM < 1e-5) = 1e-5;
    tmpValue = -temperature*boltzmann_const*log(p_GMM);
    y_val_GMM = tmpValue - min(tmpValue);
    
        
    error_GM = sqrt((y_val_GM-y_true).^2);
    error_GMM = sqrt((y_val_GMM-y_true).^2);
    
    
    y_val_kNN = interp1(dataSort,FE_kNN-min(trueFE),xVec);
    y_val_kNN(isnan(y_val_kNN)) = -temperature*boltzmann_const*log(1e-5)-min(trueFE);
    
    
    y_val_kNN_w = interp1(dataSort,FE_kNN,xVec);
    y_val_kNN_w(isnan(y_val_kNN_w)) = -temperature*boltzmann_const*log(1e-5);
    y_val_kNN_w = y_val_kNN_w - min(trueFE);
    
    error_kNN = sqrt((y_val_kNN-y_true).^2);
    
    if nMinGaussians == nMaxGaussians
        figure(1);
        
        subplot(151)
        hold on
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(edgesHist,SF_free_energy,'k-','linewidth',2);
        plot(edgesHist,SF_free_energy-SF_standard_error','k--');
        plot(edgesHist,SF_free_energy+SF_standard_error','k--');
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_hist))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(SF_standard_error))],'fontsize',12);
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('a) Histogram');
        axis square
        
        % Plot the data
        for i = 1:sampleDx:length(data)
            plot([data(i),data(i)],[3.78,3.87],'k-');
        end
        
        subplot(152)
        hold on
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(dataSort,FE_kNN-min(trueFE),'color',kNN_color,'linewidth',2);
        plot(dataSort,FE_kNN-SE_kNN-min(trueFE),'--','color',kNN_color);
        plot(dataSort,FE_kNN+SE_kNN-min(trueFE),'--','color',kNN_color);
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_kNN))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(SE_kNN))],'fontsize',12);
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('b) kNN');
        axis square
        
        % Plot the data
        for i = 1:sampleDx:length(data)
            plot([data(i),data(i)],[3.78,3.87],'k-');
        end
        
        subplot(153)
        hold on
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(x_KDE_MD,kDensityFE,'color',KDE_color,'linewidth',2);
        
        plot(x_KDE_MD,kDensityFE-stdErrorKDE,'--','color',KDE_color);
        plot(x_KDE_MD,kDensityFE+stdErrorKDE,'--','color',KDE_color);
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_KDE))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(stdErrorKDE))],'fontsize',12);
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('c) KDE');
        axis square
        
        % Plot the data
        for i = 1:sampleDx:length(data)
            plot([data(i),data(i)],[3.78,3.87],'k-');
        end
        
        subplot(154)
        hold on
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_GM))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(standardErrorGM2))],'fontsize',12);
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(minX-10,1,'k-','linewidth',2);
        plot(minX-10,1,'color',KDE_color,'linewidth',2);
        plot(dataSort,MLE_FE_GM(indSort),'color',GM_color,'linewidth',2);
        plot(dataSort,MLE_FE_GM(indSort)-standardErrorGM2(indSort)','--','color',GM_color);
        plot(dataSort,MLE_FE_GM(indSort)+standardErrorGM2(indSort)','--','color',GM_color);
        legend('True profile','Histogram','KDE','GM');
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('d) GM');
        axis square
        % Plot the data
        for i = 1:sampleDx:length(data)
            plot([data(i),data(i)],[3.78,3.87],'k-');
        end
        
        subplot(155)
        hold on
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_GMM))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(GMM_SE))],'fontsize',12);
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(minX-10,1,'k-','linewidth',2);
        plot(minX-10,1,'color',KDE_color,'linewidth',2);
        plot(minX-10,1,'color',GM_color,'linewidth',2);
        plot(dataSort,GMM_free_energy(indSort),'color',GMM_color,'linewidth',2);
        plot(dataSort,GMM_free_energy(indSort)-GMM_SE(indSort),'--','color',GMM_color);
        plot(dataSort,GMM_free_energy(indSort)+GMM_SE(indSort),'--','color',GMM_color);
        legend('True profile','Histogram','KDE','GM','GMM');
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('e) GMM');
        axis square
        % Plot the data
        for i = 1:sampleDx:length(data)
            plot([data(i),data(i)],[3.78,3.87],'k-');
        end
        
        
        figure;
        subplot(152)
        hold on
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(dataSort,FE_kNN-min(trueFE),'color',kNN_color,'linewidth',2);
        plot(dataSort,FE_kNN-SE_kNN-min(trueFE),'--','color',kNN_color);
        plot(dataSort,FE_kNN+SE_kNN-min(trueFE),'--','color',kNN_color);
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_kNN))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(SE_kNN))],'fontsize',12);
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('a) kNN');
        axis square
        
        subplot(151)
        hold on
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(edgesHist,SF_free_energy,'k-','linewidth',2);
        plot(edgesHist,SF_free_energy-SF_standard_error','k--');
        plot(edgesHist,SF_free_energy+SF_standard_error','k--');
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_hist))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(SF_standard_error))],'fontsize',12);
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('b) Histogram');
        axis square
        
        subplot(153)
        hold on
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(x_KDE_MD,kDensityFE,'color',KDE_color,'linewidth',2);
        
        plot(x_KDE_MD,kDensityFE-stdErrorKDE,'--','color',KDE_color);
        plot(x_KDE_MD,kDensityFE+stdErrorKDE,'--','color',KDE_color);
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_KDE))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(stdErrorKDE))],'fontsize',12);
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('c) KDE');
        axis square
        
        subplot(154)
        hold on
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_GM))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(standardErrorGM2))],'fontsize',12);
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(minX-10,1,'k-','linewidth',2);
        plot(minX-10,1,'color',KDE_color,'linewidth',2);
        plot(dataSort,MLE_FE_GM(indSort),'color',GM_color,'linewidth',2);
        plot(dataSort,MLE_FE_GM(indSort)-standardErrorGM2(indSort)','--','color',GM_color);
        plot(dataSort,MLE_FE_GM(indSort)+standardErrorGM2(indSort)','--','color',GM_color);
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('d) GM');
        axis square
        
        subplot(155)
        hold on
        text(0.01,0.3,['Mean true error: ', num2str(mean(error_GMM))],'fontsize',12);
        text(0.01,0.1,['Mean estimated SE: ', num2str(mean(GMM_SE))],'fontsize',12);
        plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
        plot(minX-10,1,'k-','linewidth',2);
        plot(minX-10,1,'color',KDE_color,'linewidth',2);
        plot(minX-10,1,'color',GM_color,'linewidth',2);
        plot(dataSort,GMM_free_energy(indSort),'color',GMM_color,'linewidth',2);
        plot(dataSort,GMM_free_energy(indSort)-GMM_SE(indSort),'--','color',GMM_color);
        plot(dataSort,GMM_free_energy(indSort)+GMM_SE(indSort),'--','color',GMM_color);
        legend('True profile','Histogram','KDE','GM','GMM');
        xlabel('Reaction coordinate');
        ylabel('Free energy [kcal/mol]');
        set(gca,'TickLabelInterpreter','latex');
        set(gca,'fontsize',14);
        set(gca,'xlim',[minX,maxX]);
        set(gca,'ylim',[0,4]);
        title('e) GMM');
        axis square
        
    else
        
        [MLE_FE_KDE, bestSigma, logLikelihood_MLE_KDE] = MLE_KDE_FE(data,...
            temperature, boltzmann_const, minSigma, maxSigma, data);
        
        SE_KDE = estimateKDE_SE(trajectories,...
            temperature, boltzmann_const, minSigma, maxSigma);
        
        [dataSort,indSort] = sort(data);
        
        % Estimate step function FE
         [SF_free_energy, nOptimalStepFunctions, amplitudes_SF, edges_SF, logLikelihood_MLE_SF] = MLE_SF_FE(data,...
                temperature, boltzmann_const, nMinBins, nMaxBins);
        

         [SF_standard_error, allEnergiesHist] = estimateStandardErrorStepFunction(trajectories, nMinBins, nMaxBins,...
                boltzmann_const, temperature, length(trajectories));
        
                
        % Compute error to true profile
        pTmpSF = probabilityDensityStepFunction(edges_SF, edges_SF, amplitudes_SF);
        pTmpSF = interp1(edges_SF,pTmpSF,xVec);
        pTmpSF(isnan(pTmpSF)) = 1e-5;        
        
        p_error_SF = sqrt((trueDensity-pTmpSF).^2);
        pTmpSF(pTmpSF < 1e-5) = 1e-5;
        
        FE_tmp = -temperature*boltzmann_const*log(pTmpSF);
        
        y_val_SF = FE_tmp - min(FE_tmp);
        
        p_kNN = interp1(dataSort,kNN_density,xVec);
        p_kNN(isnan(p_kNN)) = 1e-5;
        p_error_kNN = sqrt((trueDensity-p_kNN).^2);
        y_val_kNN = interp1(dataSort,FE_kNN-min(trueFE),xVec);
        y_val_kNN(isnan(y_val_kNN)) = -temperature*boltzmann_const*log(1e-5)-min(trueFE);
        
        y_true = trueFE-min(trueFE);
        
        
        [y_val_KDE, ~,~, p_KDE] = MLE_KDE_FE(data,...
            temperature, boltzmann_const, minSigma, maxSigma,xVec');
        p_error_KDE = sqrt((trueDensity-p_KDE').^2);
        
        
        error_SF = sqrt((y_val_SF-y_true).^2);
        error_GM = sqrt((y_val_GM-y_true).^2);
        error_kNN = sqrt((y_val_kNN-y_true).^2);
        error_KDE = sqrt((y_val_KDE'-y_true).^2);
        
        error_SF_all(end+1) = mean(error_SF);
        error_GM_all(end+1) = mean(error_GM);
        error_KDE_all(end+1) = mean(error_KDE);
        error_kNN_all(end+1) = mean(error_kNN);
        error_GMM_all(end+1) = mean(error_GMM);
        
        
        p_error_SF_all(end+1) = mean(p_error_SF);
        p_error_GM_all(end+1) = mean(p_error_GM);
        p_error_KDE_all(end+1) = mean(p_error_KDE);
        p_error_kNN_all(end+1) = mean(p_error_kNN);
        p_error_GMM_all(end+1) = mean(p_error_GMM);
        
        SE_SF_all(end+1) = mean(SF_standard_error);
        SE_GM_all(end+1) = mean(standardErrorGM2);
        SE_KDE_all(end+1) = mean(SE_KDE);
        SE_kNN_all(end+1) = mean(SE_kNN);
        SE_GMM_all(end+1) = mean(GMM_SE);
        
        logli_SF_all(end+1) = logLikelihood_MLE_SF;
        logli_GM_all(end+1) = logLikelihood_MLE_GM;
        logli_KDE_all(end+1) = logLikelihood_MLE_KDE;
        logli_kNN_all(end+1) = logLikelihood_kNN;
        logli_GMM_all(end+1) = GMM_logLikelihood;
    end
end
saveFolder = '/data/REMD/Plots/REST_vs_REMD_vs_MD/review2/tmp/';
fID = fopen([saveFolder,'toy_model_density_error.txt'],'w');
fprintf(fID,'SF:\n');
fprintf(fID,'%f\t', p_error_SF_all);
fprintf(fID,'\n\n');
fprintf(fID,'kNN:\n');
fprintf(fID,'%f\t', p_error_kNN_all);
fprintf(fID,'\n\n');
fprintf(fID,'KDE:\n');
fprintf(fID,'%f\t', p_error_KDE_all);
fprintf(fID,'\n\n');
fprintf(fID,'GM:\n');
fprintf(fID,'%f\t', p_error_GM_all);
fprintf(fID,'\n\n');
fprintf(fID,'GMM:\n');
fprintf(fID,'%f\t', p_error_GMM_all);
fprintf(fID,'\n\n');
fclose(fID);

fID = fopen([saveFolder,'toy_model_FE_error.txt'],'w');
fprintf(fID,'SF:\n');
fprintf(fID,'%f\t', error_SF_all);
fprintf(fID,'\n\n');
fprintf(fID,'kNN:\n');
fprintf(fID,'%f\t', error_kNN_all);
fprintf(fID,'\n\n');
fprintf(fID,'KDE:\n');
fprintf(fID,'%f\t', error_KDE_all);
fprintf(fID,'\n\n');
fprintf(fID,'GM:\n');
fprintf(fID,'%f\t', error_GM_all);
fprintf(fID,'\n\n');
fprintf(fID,'GMM:\n');
fprintf(fID,'%f\t', error_GMM_all);
fprintf(fID,'\n\n');
fclose(fID);

fID = fopen([saveFolder,'toy_model_loglikelihoods.txt'],'w');
fprintf(fID,'SF:\n');
fprintf(fID,'%f\t', logli_SF_all);
fprintf(fID,'\n\n');
fprintf(fID,'kNN:\n');
fprintf(fID,'%f\t', logli_kNN_all);
fprintf(fID,'\n\n');
fprintf(fID,'KDE:\n');
fprintf(fID,'%f\t', logli_KDE_all);
fprintf(fID,'\n\n');
fprintf(fID,'GM:\n');
fprintf(fID,'%f\t', logli_GM_all);
fprintf(fID,'\n\n');
fprintf(fID,'GMM:\n');
fprintf(fID,'%f\t', logli_GMM_all);
fprintf(fID,'\n\n');
fclose(fID);

fID = fopen([saveFolder,'toy_model_SE.txt'],'w');
fprintf(fID,'SF:\n');
fprintf(fID,'%f\t', SE_SF_all);
fprintf(fID,'\n\n');
fprintf(fID,'kNN:\n');
fprintf(fID,'%f\t', SE_kNN_all);
fprintf(fID,'\n\n');
fprintf(fID,'KDE:\n');
fprintf(fID,'%f\t', SE_KDE_all);
fprintf(fID,'\n\n');
fprintf(fID,'GM:\n');
fprintf(fID,'%f\t', SE_GM_all);
fprintf(fID,'\n\n');
fprintf(fID,'GMM:\n');
fprintf(fID,'%f\t', SE_GMM_all);
fprintf(fID,'\n\n');
fclose(fID);

text_y1 = -0.5;
text_y2 = -0.25;
text_y3 = 0.0;
text_y4 = -0.75;
y_cutoff = 0; %-0.85;
y_cutoff2 = 4;


if nMinGaussians ~= nMaxGaussians
    figure;
    subplot(151)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(dataSort,SF_free_energy(indSort),'k-','linewidth',2);
    plot(dataSort,SF_free_energy(indSort)-SF_standard_error(indSort)','k--');
    plot(dataSort,SF_free_energy(indSort)+SF_standard_error(indSort)','k--');
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_SF))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_SF))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(SF_standard_error))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(logLikelihood_MLE_SF)],'fontsize',12);
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('a)');
    % Plot the data
    %     for i = 1:sampleDx:length(dataSort)
    %         plot([dataSort(i),dataSort(i)],[3.78,3.87],'k-');
    %     end
    
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(trajectories)
        tmpData = trajectories{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    axis square
    
    subplot(152)
    hold on
    
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',GM_color,'linewidth',2);
    plot(dataSort,FE_kNN-min(trueFE),'color',kNN_color,'linewidth',2);
    plot(dataSort,FE_kNN-SE_kNN-min(trueFE),'--','color',kNN_color);
    plot(dataSort,FE_kNN+SE_kNN-min(trueFE),'--','color',kNN_color);
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_kNN))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_kNN))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(SE_kNN))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(logLikelihood_kNN)],'fontsize',12);
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('b)');
    % Plot the data
    %     for i = 1:sampleDx:length(dataSort)
    %         plot([dataSort(i),dataSort(i)],[3.78,3.87],'k-');
    %     end
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(trajectories)
        tmpData = trajectories{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    axis square
    
    subplot(153)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(dataSort,MLE_FE_KDE(indSort),'color',KDE_color,'linewidth',2);
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_KDE))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_KDE))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(SE_KDE))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(logLikelihood_MLE_KDE)],'fontsize',12);
    plot(dataSort,MLE_FE_KDE(indSort)+SE_KDE(indSort),'--','color',KDE_color);
    plot(dataSort,MLE_FE_KDE(indSort)-SE_KDE(indSort),'--','color',KDE_color);
    
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('d)');
    
    % Plot the data
    %     for i = 1:sampleDx:length(dataSort)
    %         plot([dataSort(i),dataSort(i)],[3.78,3.87],'k-');
    %     end
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(trajectories)
        tmpData = trajectories{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    axis square
    
    
    subplot(154)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(dataSort,MLE_FE_GM(indSort),'color',GM_color,'linewidth',2);
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_GM))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_GM))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(standardErrorGM2))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(logLikelihood_MLE_GM)],'fontsize',12);
    plot(dataSort,MLE_FE_GM(indSort)+standardErrorGM2(indSort)','--','color',GM_color);
    plot(dataSort,MLE_FE_GM(indSort)-standardErrorGM2(indSort)','--','color',GM_color);
    
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('d)');
    
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(trajectories)
        tmpData = trajectories{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    
    axis square
    
    subplot(155)
    hold on
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_GMM))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_GMM))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(GMM_SE))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(GMM_logLikelihood)],'fontsize',12);
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(minX-10,1,'color',KDE_color,'linewidth',2);
    plot(minX-10,1,'color',GM_color,'linewidth',2);
    plot(dataSort,GMM_free_energy(indSort),'color',GMM_color,'linewidth',2);
    plot(dataSort,GMM_free_energy(indSort)-GMM_SE(indSort),'--','color',GMM_color);
    plot(dataSort,GMM_free_energy(indSort)+GMM_SE(indSort),'--','color',GMM_color);
    legend('True profile','Histogram','kNN','KDE','GM','GMM');
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('e) GMM');
    axis square
    
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(trajectories)
        tmpData = trajectories{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    
    set(gcf,'renderer','painter')
    
    figure;
    subplot(151)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(dataSort,SF_free_energy(indSort),'k-','linewidth',2);
    plot(dataSort,SF_free_energy(indSort)-SF_standard_error(indSort)','k--');
    plot(dataSort,SF_free_energy(indSort)+SF_standard_error(indSort)','k--');
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('a) Step function');
    axis square
    
    subplot(152)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',GM_color,'linewidth',2);
    plot(dataSort,FE_kNN-min(trueFE),'color',kNN_color,'linewidth',2);
    plot(dataSort,FE_kNN-SE_kNN-min(trueFE),'--','color',kNN_color);
    plot(dataSort,FE_kNN+SE_kNN-min(trueFE),'--','color',kNN_color);
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('b) kNN');
    axis square
    
    subplot(153)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(dataSort,MLE_FE_KDE(indSort),'color',KDE_color,'linewidth',2);
    plot(dataSort,MLE_FE_KDE(indSort)+SE_KDE(indSort),'--','color',KDE_color);
    plot(dataSort,MLE_FE_KDE(indSort)-SE_KDE(indSort),'--','color',KDE_color);
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('c) KDE');
    axis square
    
    
    subplot(154)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(dataSort,MLE_FE_GM(indSort),'color',GM_color,'linewidth',2);
    plot(dataSort,MLE_FE_GM(indSort)+standardErrorGM2(indSort)','--','color',GM_color);
    plot(dataSort,MLE_FE_GM(indSort)-standardErrorGM2(indSort)','--','color',GM_color);
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('d) GM');
    axis square
    
    subplot(155)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(minX-10,1,'color',KDE_color,'linewidth',2);
    plot(minX-10,1,'color',GM_color,'linewidth',2);
    plot(dataSort,GMM_free_energy(indSort),'color',GMM_color,'linewidth',2);
    plot(dataSort,GMM_free_energy(indSort)-GMM_SE(indSort),'--','color',GMM_color);
    plot(dataSort,GMM_free_energy(indSort)+GMM_SE(indSort),'--','color',GMM_color);
    legend('True profile','Histogram','kNN','KDE','GM','GMM');
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('e) GMM');
    axis square
    set(gcf,'renderer','painter')
    
elseif false
    figure;
    subplot(151)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(dataSort,averageEnergy(indSort),'k-','linewidth',2);
    plot(dataSort,averageEnergy(indSort)-stErrEnergies1(indSort)','k--');
    plot(dataSort,averageEnergy(indSort)+stErrEnergies1(indSort)','k--');
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_SF_all))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_SF_all))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(SE_SF_all))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(mean(logli_SF_all))],'fontsize',12);
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('a) Step function');
    
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(tmpEdges)
        tmpData = tmpEdges{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    axis square
    
    subplot(152)
    hold on
    
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',GM_color,'linewidth',2);
    plot(dataSort,FE_kNN-min(trueFE),'color',kNN_color,'linewidth',2);
    plot(dataSort,FE_kNN-SE_kNN-min(trueFE),'--','color',kNN_color);
    plot(dataSort,FE_kNN+SE_kNN-min(trueFE),'--','color',kNN_color);
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_kNN_all))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_kNN_all))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(SE_kNN_all))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(mean(logli_kNN_all))],'fontsize',12);
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('b) kNN');

    
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(tmpEdges)
        tmpData = tmpEdges{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    axis square
    
    subplot(153)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(dataSort,MLE_FE_KDE(indSort),'color',KDE_color,'linewidth',2);
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_KDE_all))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_KDE_all))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(SE_KDE_all))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(mean(logli_KDE_all))],'fontsize',12);
    plot(dataSort,MLE_FE_KDE(indSort)+SE_KDE(indSort),'--','color',KDE_color);
    plot(dataSort,MLE_FE_KDE(indSort)-SE_KDE(indSort),'--','color',KDE_color);
    
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('c) KDE');

    y_vals = 3.08:0.1:3.98;
    for j = 1:length(tmpEdges)
        tmpData = tmpEdges{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    axis square
    
    
    subplot(154)
    hold on
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(dataSort,MLE_FE_GM(indSort),'color',GM_color,'linewidth',2);
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_GM_all))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_GM_all))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(SE_GM_all))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(mean(logli_GM_all))],'fontsize',12);
    plot(dataSort,MLE_FE_GM(indSort)+standardErrorGM2(indSort)','--','color',GM_color);
    plot(dataSort,MLE_FE_GM(indSort)-standardErrorGM2(indSort)','--','color',GM_color);
    
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('d) GM grid');
    
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(tmpEdges)
        tmpData = tmpEdges{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    
    axis square
    
    subplot(155)
    hold on
    text(0.01,text_y1,['Mean true error FE: ', num2str(mean(error_GMM_all))],'fontsize',12);
    text(0.01,text_y4,['Mean true error density: ', num2str(mean(p_error_GMM_all))],'fontsize',12);
    text(0.01,text_y2,['Mean estimated SE: ', num2str(mean(SE_GMM_all))],'fontsize',12);
    text(0.01,text_y3,['Log-likelihood: ', num2str(mean(logli_GMM_all))],'fontsize',12);
    plot(xVec,trueFE-min(trueFE),'-','color',trueColor,'markerfacecolor',trueColor,'linewidth',2);
    plot(minX-10,1,'k-','linewidth',2);
    plot(minX-10,1,'color',kNN_color,'linewidth',2);
    plot(minX-10,1,'color',KDE_color,'linewidth',2);
    plot(minX-10,1,'color',GM_color,'linewidth',2);
    plot(dataSort,GMM_free_energy(indSort),'color',GMM_color,'linewidth',2);
    plot(dataSort,GMM_free_energy(indSort)-GMM_SE(indSort),'--','color',GMM_color);
    plot(dataSort,GMM_free_energy(indSort)+GMM_SE(indSort),'--','color',GMM_color);
    legend('True profile','Histogram','kNN','KDE','GM','GMM');
    xlabel('Reaction coordinate');
    ylabel('Free energy [kcal/mol]');
    set(gca,'TickLabelInterpreter','latex');
    set(gca,'fontsize',14);
    set(gca,'xlim',[minX,maxX]);
    set(gca,'ylim',[y_cutoff,y_cutoff2])
    title('e) GMM');
    axis square
    
    y_vals = 3.08:0.1:3.98;
    for j = 1:length(tmpEdges)
        tmpData = tmpEdges{j};
        for i = 1:sampleDx:length(tmpData)
            plot([tmpData(i),tmpData(i)],[y_vals(j)-0.09,y_vals(j)],'k-');
        end
    end
    
    set(gcf,'renderer','painter')
    
end


nRuns = length(SE_SF_all);

figure;
set(gcf,'renderer','painter');

subplot(1,4,1)
set(gca,'ticklabelinterpreter','latex');

plot(1.0:5.0,[mean(SE_SF_all),mean(SE_kNN_all),mean(SE_KDE_all),...
    mean(SE_GM_all),mean(SE_GMM_all)],'--','linewidth',2,'color',[0,0,0]);
hold on

plot(1.0*ones(nRuns),SE_SF_all,'.','color',SF_color);
plot(2.0*ones(nRuns),SE_kNN_all,'.','color',kNN_color);
plot(3.0*ones(nRuns),SE_KDE_all,'.','color',KDE_color);
plot(4.0*ones(nRuns),SE_GM_all,'.','color',GM_color);
plot(5.0*ones(nRuns),SE_GMM_all,'.','color',GMM_color);

plot(1.0,mean(SE_SF_all),'o','color',SF_color,'markerfacecolor',SF_color);
plot(2.0,mean(SE_kNN_all),'o','color',kNN_color,'markerfacecolor',kNN_color);
plot(3.0,mean(SE_KDE_all),'o','color',KDE_color,'markerfacecolor',KDE_color);
plot(4.0,mean(SE_GM_all),'o','color',GM_color,'markerfacecolor',GM_color);
plot(5.0,mean(SE_GMM_all),'o','color',GMM_color,'markerfacecolor',GMM_color);

set(gca,'xlim',[0.5,5.5]);
set(gca,'ticklabelinterpreter','latex','fontsize',13);
ylabel('Standard error');

set(gca,'xtick',1:5);
set(gca,'xticklabel',{'Step function','kNN','KDE','GM','GMM'});
title('f)');

subplot(1,4,2)
set(gca,'ticklabelinterpreter','latex','fontsize',13);


hold on
plot(1.0:5.0,[mean(error_SF_all),mean(error_kNN_all),mean(error_KDE_all),...
    mean(error_GM_all),mean(error_GMM_all)],'--','color',[0,0,0],'linewidth',2);

plot(1.0*ones(nRuns),error_SF_all,'.','color',SF_color);
plot(2.0*ones(nRuns),error_kNN_all,'.','color',kNN_color);
plot(3.0*ones(nRuns),error_KDE_all,'.','color',KDE_color);
plot(4.0*ones(nRuns),error_GM_all,'.','color',GM_color);
plot(5.0*ones(nRuns),error_GMM_all,'.','color',GMM_color);

plot(1.0,mean(error_SF_all),'o','color',SF_color,'markerfacecolor',SF_color);
plot(2.0,mean(error_kNN_all),'o','color',kNN_color,'markerfacecolor',kNN_color);
plot(3.0,mean(error_KDE_all),'o','color',KDE_color,'markerfacecolor',KDE_color);
plot(4.0,mean(error_GM_all),'o','color',GM_color,'markerfacecolor',GM_color);
plot(5.0,mean(error_GMM_all),'o','color',GMM_color,'markerfacecolor',GMM_color);

set(gca,'ticklabelinterpreter','latex','fontsize',13);
ylabel('True FE error');
set(gca,'xlim',[0.5,5.5]);

set(gca,'xtick',1:5);
set(gca,'xticklabel',{'Step function','kNN','KDE','GM','GMM'});
title('g)');

subplot(1,4,3)
set(gca,'ticklabelinterpreter','latex','fontsize',13);


hold on
plot(1.0:5.0,[mean(p_error_SF_all),mean(p_error_kNN_all),mean(p_error_KDE_all),...
    mean(p_error_GM_all),mean(p_error_GMM_all)],'--','color',[0,0,0],'linewidth',2);

plot(1.0*ones(nRuns),p_error_SF_all,'.','color',SF_color);
plot(2.0*ones(nRuns),p_error_kNN_all,'.','color',kNN_color);
plot(3.0*ones(nRuns),p_error_KDE_all,'.','color',KDE_color);
plot(4.0*ones(nRuns),p_error_GM_all,'.','color',GM_color);
plot(5.0*ones(nRuns),p_error_GMM_all,'.','color',GMM_color);

plot(1.0,mean(p_error_SF_all),'o','color',SF_color,'markerfacecolor',SF_color);
plot(2.0,mean(p_error_kNN_all),'o','color',kNN_color,'markerfacecolor',kNN_color);
plot(3.0,mean(p_error_KDE_all),'o','color',KDE_color,'markerfacecolor',KDE_color);
plot(4.0,mean(p_error_GM_all),'o','color',GM_color,'markerfacecolor',GM_color);
plot(5.0,mean(p_error_GMM_all),'o','color',GMM_color,'markerfacecolor',GMM_color);

set(gca,'ticklabelinterpreter','latex','fontsize',13);
ylabel('True density error');
set(gca,'xlim',[0.5,5.5]);

set(gca,'xtick',1:5);
set(gca,'xticklabel',{'Step function','kNN','KDE','GM','GMM'});
title('h)');

subplot(1,4,4)
plot(1:5,[mean(logli_SF_all),mean(logli_kNN_all),mean(logli_KDE_all),...
    mean(logli_GM_all),mean(logli_GMM_all)],'--','linewidth',2,'color',[0,0,0]);
hold on

plot(1*ones(nRuns),logli_SF_all,'.','color',SF_color);
plot(2*ones(nRuns),logli_kNN_all,'.','color',kNN_color);
plot(3*ones(nRuns),logli_KDE_all,'.','color',KDE_color);
plot(4*ones(nRuns),logli_GM_all,'.','color',GM_color);
plot(5*ones(nRuns),logli_GMM_all,'.','color',GMM_color);

plot(1,mean(logli_SF_all),'o','color',SF_color,'markerfacecolor',SF_color);
plot(2,mean(logli_kNN_all),'o','color',kNN_color,'markerfacecolor',kNN_color);
plot(3,mean(logli_KDE_all),'o','color',KDE_color,'markerfacecolor',KDE_color);
plot(4,mean(logli_GM_all),'o','color',GM_color,'markerfacecolor',GM_color);
plot(5,mean(logli_GMM_all),'o','color',GMM_color,'markerfacecolor',GMM_color);

set(gca,'xtick',1:5);
set(gca,'xticklabel',{'Step function','kNN','KDE','GM','GMM'});
set(gca,'xlim',[0.5,5.5]);
set(gca,'ticklabelinterpreter','latex','fontsize',13);
ylabel('Log-likelihood');
title('i)');



figure;
subplot(1,4,1)
hold on
plot(logli_SF_all,'ko-','markerfacecolor','k');
plot(logli_kNN_all,'-o','color',kNN_color,'markerfacecolor',kNN_color);
plot(logli_KDE_all,'-o','color',KDE_color,'markerfacecolor',KDE_color);
plot(logli_GM_all,'-o','color',GM_color,'markerfacecolor',GM_color);
plot(logli_GMM_all,'-o','color',GMM_color,'markerfacecolor',GMM_color);
ylabel('Log-likelihood')
xlabel('Simulation number')
legend(['SF mean: ',num2str(mean(logli_SF_all))], ['kNN mean: ',num2str(mean(logli_kNN_all))],...
    ['KDE mean: ',num2str(mean(logli_KDE_all))], ['GM mean: ',num2str(mean(logli_GM_all))],...
    ['GMM mean: ',num2str(mean(logli_GMM_all))])

subplot(1,4,2)
hold on
plot(SE_SF_all,'ko-','markerfacecolor','k');
plot(SE_kNN_all,'-o','color',kNN_color,'markerfacecolor',kNN_color);
plot(SE_KDE_all,'-o','color',KDE_color,'markerfacecolor',KDE_color);
plot(SE_GM_all,'-o','color',GM_color,'markerfacecolor',GM_color);
plot(SE_GMM_all,'-o','color',GMM_color,'markerfacecolor',GMM_color);
xlabel('Simulation number')

ylabel('Standard error')
legend(['SF mean: ',num2str(mean(SE_SF_all))], ['kNN mean: ',num2str(mean(SE_kNN_all))],...
    ['KDE mean: ',num2str(mean(SE_KDE_all))], ['GM mean: ',num2str(mean(SE_GM_all))],...
    ['GMM mean: ',num2str(mean(SE_GMM_all))])

subplot(1,4,3)
hold on
plot(error_SF_all,'ko-','markerfacecolor','k');
plot(error_kNN_all,'-o','color',kNN_color,'markerfacecolor',kNN_color);
plot(error_KDE_all,'-o','color',KDE_color,'markerfacecolor',KDE_color);
plot(error_GM_all,'-o','color',GM_color,'markerfacecolor',GM_color);
plot(error_GMM_all,'-o','color',GMM_color,'markerfacecolor',GMM_color);
xlabel('Simulation number')

ylabel('FE error')
legend(['SF mean: ',num2str(mean(error_SF_all))], ['kNN mean: ',num2str(mean(error_kNN_all))],...
    ['KDE mean: ',num2str(mean(error_KDE_all))], ['GM mean: ',num2str(mean(error_GM_all))],...
    ['GMM mean: ',num2str(mean(error_GMM_all))])

subplot(1,4,4)
hold on
plot(p_error_SF_all,'ko-','markerfacecolor','k');
plot(p_error_kNN_all,'-o','color',kNN_color,'markerfacecolor',kNN_color);
plot(p_error_KDE_all,'-o','color',KDE_color,'markerfacecolor',KDE_color);
plot(p_error_GM_all,'-o','color',GM_color,'markerfacecolor',GM_color);
plot(p_error_GMM_all,'-o','color',GMM_color,'markerfacecolor',GMM_color);
xlabel('Simulation number')

ylabel('Density error')
legend(['SF mean: ',num2str(mean(p_error_SF_all))], ['kNN mean: ',num2str(mean(p_error_kNN_all))],...
    ['KDE mean: ',num2str(mean(p_error_KDE_all))], ['GM mean: ',num2str(mean(p_error_GM_all))],...
    ['GMM mean: ',num2str(mean(p_error_GMM_all))])

    set(gcf,'renderer','painter')
