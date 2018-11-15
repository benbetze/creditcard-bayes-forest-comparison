%% Import Data
    clear all, clc
    close all
    input = csvread('UCI_Credit_Card.csv',1,1);
    name = {'LIMIT_BAL' 'SEX' 'EDUCATION' 'MARRIAGE' 'AGE' 'PAY_0' 'PAY_2' 'PAY_3' 'PAY_4' 'PAY_5' 'PAY_6' 'BILL_AMT1' 'BILL_AMT2' 'BILL_AMT3' 'BILL_AMT4' 'BILL_AMT5' 'BILL_AMT6' 'PAY_AMT1' 'PAY_AMT2' 'PAY_AMT3' 'PAY_AMT4' 'PAY_AMT5' 'PAY_AMT6' 'default'};
    name_continous = {'LIMIT_BAL' 'AGE' 'PAY_0' 'PAY_2' 'PAY_3' 'PAY_4' 'PAY_5' 'PAY_6' 'BILL_AMT1' 'BILL_AMT2' 'BILL_AMT3' 'BILL_AMT4' 'BILL_AMT5' 'BILL_AMT6' 'PAY_AMT1' 'PAY_AMT2' 'PAY_AMT3' 'PAY_AMT4' 'PAY_AMT5' 'PAY_AMT6'};
    data = cell2table(num2cell(input),'VariableNames',name);
%% Check Data
    TF = ismissing(data);
    CheckTF = sum(TF);
        if CheckTF == 0
            disp('Everything is ok, no missing data')
        else
            disp('Missing data!')
        end
% Delete TF and CheckTF because we do not need them anymore
    clear TF CheckTF;
%% Transform Data
% Define categorical data
    valueset_catDefault = [0:1];
    catnames_default = {'non-default', 'default'};
        data.default = categorical(data.default, valueset_catDefault, catnames_default, 'Ordinal', true);
    valueset_catSEX = (1:2);
    catnames_SEX = {'male', 'female'};
        data.SEX = categorical(data.SEX, valueset_catSEX, catnames_SEX,'Ordinal', true);
    valueset_catEDUCATION = [1:4];
    catnames_EDUCATION = {'graduate school', 'university', 'high school', 'others'};
        data.EDUCATION = categorical(data.EDUCATION, valueset_catEDUCATION, catnames_EDUCATION, 'Ordinal', true);
    valueset_catMARRIAGE = [1:3];
    catnames_MARRIAGE = {'married', 'single', 'others'};
        data.MARRIAGE = categorical(data.MARRIAGE, valueset_catMARRIAGE, catnames_MARRIAGE, 'Ordinal', true);
% Delete variables which are not needed any more
    clear valueset_catDefault   valueset_catDefault catnames_default valueset_catSEX catnames_SEX valueset_catEDUCATION catnames_EDUCATION valueset_catMARRIAGE catnames_MARRIAGE;
% Display amount of categorical / continous variables
    disp(['This dataset has ',num2str(sum(varfun(@isnumeric,data,'output','uniform'))),' continous variables ' ...
        'and ' num2str(24 - sum(varfun(@isnumeric,data,'output','uniform'))) ' categorical variables'])
% Only numeric table
    dataNumeric = table2array(data(:,varfun(@isnumeric,data,'output','uniform')));
% Standardize Numerical data for Naive Bayes
    dataStandardized = data;
    for i = 5:23
        columnToStandardize = dataStandardized(:,i);
        columnStandardized = zscore(table2array(columnToStandardize(:,1)));
        dataStandardized(:,i) = array2table(columnStandardized);
    end
%% Basic statistics 
    summaryStats = summary(data);
    groupAge = grpstats(data.AGE, data.default);
    mean(data.AGE);
    mean(data.LIMIT_BAL);

    dataStat = data(:,{'LIMIT_BAL','AGE','default'});
    statarray = grpstats(dataStat,'default')
    
    stats_cat = grpstats(data,{'default'},{'min','max','mean','std', 'gname'},'DataVars',{'AGE','LIMIT_BAL'})   
% Distribution
% - Age
    [KerDis,Status] = fitdist(data.AGE,'Kernel','by',data.default)
    NonDefault = KerDis{1}
    Default = KerDis{2}

    x = 0:1:100;
    NonDefault_pdf = pdf(NonDefault,x);
    Default_pdf = pdf(Default,x);
    figure
    plot(x,NonDefault_pdf,'r-')
    hold on
    plot(x,Default_pdf,'b-.')
    legend({'Non-Default','Default'},'Location','NW')
    title('Age by Default Status')
    xlabel('Age')
    hold off
% - Limit
    [KerDis,Status] = fitdist(data.LIMIT_BAL,'Kernel','by',data.default)
    NonDefault = KerDis{1}
    Default = KerDis{2}

    x = 1000:100:1000000;
    NonDefault_pdf = pdf(NonDefault,x);
    Default_pdf = pdf(Default,x);
    figure
    plot(x,NonDefault_pdf,'r-')
    hold on
    plot(x,Default_pdf,'b-.')
    legend({'Non-Default','Default'},'Location','NW')
    title('Limit balance by Default Status')
    xlabel('Limit Balance')
    hold off
% Histogram
% Age
    figure
    hold on
    histogram(data.AGE(data.default=='non-default'))
    histogram(data.AGE(data.default=='default'))
    legend({'Non-default','Default'})
    title('Age by default status')
% Education
    figure
    hold on
    histogram(data.EDUCATION(data.default=='non-default'))
    histogram(data.EDUCATION(data.default=='default'))
    legend({'Non-default','Default'})
    title('Education by default status')
% Sex
    figure
    hold on
    histogram(data.SEX(data.default=='non-default'))
    histogram(data.SEX(data.default=='default'))
    legend({'Non-default','Default'})
    title('Gender by default status')  
%% Split Data
    cvpt = cvpartition(data.default,'HoldOut',0.90);
    trainData = data(training(cvpt),:);
    trainDataSt = dataStandardized(training(cvpt),:);
    testData = data(test(cvpt),:);
    testDataSt = dataStandardized(test(cvpt),:);
    idxTrain = training(cvpt);
%% Classification
% % Random Forest
%     maxMinLS = 20;
%     minLS = optimizableVariable('MinLeafSize',[1,maxMinLS],'Type','integer');
%     numSPL = optimizableVariable('MaxNumSplits',[1,size(trainData,2)-1],'Type','integer');
%     splitCrit = optimizableVariable('SplitCriterion',{'gdi'  'deviance'});
%     hyperparametersRF = [minLS;numSPL;splitCrit];
% 
%     mdlRF = fitcensemble(trainData,'default','Method','Bag','Learner','Tree','NumLearningCycles',100,'OptimizeHyperparameters',hyperparametersRF,'HyperparameterOptimizationOptions',struct('Optimizer','randomsearch'))
% 
% % Show Training Error - Random Forests
% 
%     % Measure the training error (resubstitution loss) using the function
%     % resubLoss. You can use the resubLoss function to measure the predictive inaccuracy of the model on the training data.
% 
%     trainErrRF = resubLoss(mdlRF);
%         disp(['Training Error Random Forests: ',num2str(trainErrRF)])
%     oobLoss(mdlRF)

% Naive Bayes

    mdlNB = fitcsvm(trainData,'default')
    
% Show Training Error - Naive Bayes
    trainErrNB = resubLoss(mdlNB);
        disp(['Training Error Naive Bayes: ',num2str(trainErrNB)])

% Alternative Naive Bayes models

    % mdlNB = fitcnb(trainDataSt,'default','OptimizeHyperparameters','auto',...
    %  'HyperparameterOptimizationOptions',struct('Optimizer',...
    % 'randomsearch'))

    %mdlNB2 = fitcnb(trainDataSt,'default','OptimizeHyperparameters','auto')

    %mdlNB = fitcnb(trainData,'default')

% Predict
    predRF = predict(mdlRF,testData); % Random Forests
    predNB = predict(mdlNB,testDataSt); % Naive Bayes
%% Accuracy and loss results
% Use the function loss to calculate the test or validation error
    lossNB = loss(mdlNB,testDataSt);
    lossRF = loss(mdlRF,testData);
    accNB = 1 - lossNB;
        disp(['Accuracy NB: ',num2str(accNB)])
    accRF = 1 - lossRF;
        disp(['Accuracy RF: ',num2str(accNB)])
%% Create a heatmap for confussion matrix    
% Naive Bayes heatmap
    comatNB = confusionmat(testDataSt.default,predNB);
    [cm,grp] = confusionmat(testDataSt.default,predNB);

    figure();
    heatmap(grp,grp,cm);
    title('Heatmap Confusion Matrix - NB')

    falseNeg = 100*comatNB(1,2)/sum(sum(comatNB));
    falsePos = 100*comatNB(2,1)/sum(sum(comatNB));

        disp(['Accuracy NB: ',num2str(accNB),'%'])
        disp(['Percentage of False Negatives NB: ',num2str(falseNeg),'%'])
        disp(['Percentage of False Positives NB: ',num2str(falsePos),'%'])
% Random Forests heatmap
    comatRF = confusionmat(testData.default,predRF);
    [cm,grp] = confusionmat(testData.default,predRF);
    figure();
    heatmap(grp,grp,cm);
    title('Heatmap Confusion Matrix - RF')

    falseNeg = 100*comatRF(1,2)/sum(sum(comatRF));
    falsePos = 100*comatRF(2,1)/sum(sum(comatRF));

        disp(['Accuracy RF: ',num2str(accRF),'%'])
        disp(['Percentage of False Negatives RF: ',num2str(falseNeg),'%'])
        disp(['Percentage of False Positives RF: ',num2str(falsePos),'%'])
%% Compare
% Simple graph
    compare = [lossNB, lossRF ; trainErrNB, trainErrRF];
    figure();
    bar(compare) 
    set(gca,'xticklabel',{'NB'; 'RF'});
    ylim([0 1])
    legend('resubLoss','loss');
% ROC
% Naive Bayes
    [~,scoreNB] = resubPredict(mdlNB);
    diffscoreNB = scoreNB(:,2);
    [Xnb,Ynb,Tnb,AUCnb] = perfcurve(trainDataSt.default,diffscoreNB,'default')
    figure
    plot(Xnb,Ynb)
    hold on
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('ROC Curve for Classification - NB')
    hold off
% Random Forests
    [~,scoreRF] = resubPredict(mdlRF);
    diffscoreRF = scoreRF(:,2);
    [Xrf,Yrf,Trf,AUCrf] = perfcurve(trainData.default,diffscoreRF,'default')
    figure
    plot(Xrf,Yrf)
    hold on
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('ROC Curve for Classification - RF')
    hold off