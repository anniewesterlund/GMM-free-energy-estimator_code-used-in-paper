function [averageEnergy, stdevEnergy1, stdevEnergy2, edges, allEnergies, minEnergy] = histogram_FE(orderParameterValues, kb, T, nBins, nBootstrapResamples)

% Compute average free energy given a cell array with order parameter
% values.

nSimulations = length(orderParameterValues);
zeroCompensation = 1e-5;


allOrderparameterValues = orderParameterValues{1};
for i = 2:nSimulations
    allOrderparameterValues = [allOrderparameterValues; orderParameterValues{i}];
end

[averageHist, edges] = hist(allOrderparameterValues,nBins);
averageHist = averageHist./trapz(edges,averageHist);
averageHist(averageHist < zeroCompensation) = zeroCompensation;

% [standardError, allEnergies] = bootstrapEnergyStandardErrorNP(allOrderparameterValues,...
%     kb, T, nBootstrapResamples, nBins);
% 
% stdevEnergy2 = standardError;
% stdevEnergy1 = standardError;

averageEnergy = computeFreeEnergyFromHistogram(kb, T,averageHist); %-kb.*T.*log(averageHist(averageHist ~= 0));

minEnergy = min(averageEnergy);
averageEnergy = averageEnergy - min(averageEnergy);

if true
    histMat = zeros(nSimulations,length(edges));
    
    for i = 1:nSimulations
        hTmp = hist(orderParameterValues{i},edges);
        tmpPdf = hTmp./trapz(edges,hTmp);
        
        tmpPdf(tmpPdf < zeroCompensation) = zeroCompensation;
        tmpPdf(isnan(tmpPdf)) = zeroCompensation;
        histMat(i,:) = tmpPdf;
    end
    
    % figure(13);
    % plot(averageHist);
    % hold all
    % plot(lowerBound);
    % plot(upperBound);
    % hold off
    % lowerBound = averageHist - varHist;
    % upperBound = averageHist + varHist;
    
    % averageEnergy(averageHist == 0) = nan; %-kb.*T.*log(0.001);
    
    %%% OLD WAY OFF COMPUTING THE ERRORS FROM HISTOGRAMS and propagate through
    %%% energy. Removed now.
    % for i = 1:size(histMat,2)
    %     tmpState = histMat(:,i);
    % %     averageHist(i) = mean(tmpState(~isnan(tmpState))); 1e-7;
    %     varHist(i) = var(tmpState(~isnan(tmpState)));
    % end
    % lowerBound = averageHist - sqrt(varHist./(size(histMat,1)));
    % lowerBound(lowerBound < 0) = 0.0001;
    % upperBound = averageHist + sqrt(varHist./(size(histMat,1)));
    % stdevEnergy1(upperBound ~= 0) = computeFreeEnergyFromHistogram(kb, T, ...
    %     upperBound(upperBound ~= 0));
    % stdevEnergy1(upperBound == 0) = nan;
    %
    % stdevEnergy2(lowerBound ~= 0) = computeFreeEnergyFromHistogram(kb, T, ...
    %     lowerBound(lowerBound ~= 0));
    % stdevEnergy2(lowerBound == 0) = nan;
    %
    %%%%%%%%%%%%%%%%%%%
    
    allEnergies = cell(size(histMat,1),1);
    allEnerMat = zeros(size(histMat));
    for i = 1:size(histMat,1)
        allEnergies{i} = zeros(size(histMat,2),1);
        allEnergies{i} = computeFreeEnergyFromHistogram(kb, T, histMat(i,:))';
        allEnerMat(i,:) = allEnergies{i};
    end
    
    stdevEnergy1 = zeros(size(allEnerMat,2),1);
    for i = 1:size(allEnerMat,2)
        tmpE = allEnerMat(:,i); %allEnerMat(~isnan(allEnerMat(:,i)),i);
        stdevEnergy1(i) = sqrt(var(tmpE)/length(tmpE));
    end
    
    stdevEnergy2 = stdevEnergy1;
end

end
