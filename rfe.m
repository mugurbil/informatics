function [features,perf] = rfe(data,target,train_indx,test_indx,reduction_factor,perform_stat_comparison,classifier)
% RFE Recursive feature elimination.
%
% INPUTS
% [1] DATA        : dataset without target
% [2] TARGET      : vector of target class values
% [3] TRAIN_INDX  : training indices
% [4] TEST_INDX   : testing indices
% [5] REDUCTION_FACTOR : each recursion, the algorithm will divide the
%     number of features by this factor
% [6] PERFORM_STAT_COMPARISON
% [7] CLASSIFIER  : linear, svm or rf
% 
% OUTPUTS
% [1] FEATURES    : rfe selected feature set
% [2] PERF        : auc of the classification using the selected features
% 
% COMMENTS
% [1] Compute weights for the features
% [2] Train SVM model, evaluate its performance, and compute weights
% [3] Determine the minimal number of features yeilding the best performance
% [4] Perform statistical comparison with the best performing feature set
% [5] Return features

% LOG
% [2.1] (Dec 29, 2017) Dependencies from auc and compare_auc to fauc.
% [2.0] (Nov 28, 2017) Joined svm and random forest implementations.
% [1.2] (Nov 14, 2017) Random forest implementation. - Sisi Ma
% [1.1] (Oct 31, 2017) Automatic svm kernel selection. Modernize.
% [1.0] - DSL
%
% Developed by Mehmet Ugurbil

% Copyright (C) 2017-2018 University of Minnesota

if nargin < 7; classifier = 'svm'; end
if nargin < 6; perform_stat_comparison = 1; end
if nargin < 5; reduction_factor = 2.; end

[~, n_vars] = size(data);

N_features = unique(round(exp(log(n_vars):-log(reduction_factor):0)), 'stable');
len = length(N_features);

P = zeros(len, 1);
prediction = cell(len, 1);
features = cell(len, 1);

for i = 1:len
    if i == 1
        features{i} = 1:n_vars;
    else
        [~, indx] = sort(weights, 'descend');
        sorted_previous_features = features{i-1}(indx);
        features{i} = sorted_previous_features(1:N_features(i));
    end
    [P(i), prediction{i}, weights] = classification(data(:,features{i}),target,train_indx,test_indx,classifier);
end

indx = find( (max(P) - P) <= 2*eps, 1, 'last');
if perform_stat_comparison
    pv = zeros(len,1);
    for i = 1:len
        if i == indx || P(i) == P(indx)
            pv(i) = 1;
        else
            pv(i) = fauc(target(test_indx), prediction{i}, prediction{indx});
        end
    end
    indx = find( pv > 0.05 , 1, 'last' );
end

perf = P(indx);
features = features{indx};

function [perf,prediction,weights] = classification(data,target,train_indx,test_indx,classifier)

switch upper(classifier)
    case 'LINEAR'
        model = fitclinear(data(train_indx,:), target(train_indx));
        prediction = model.Bias + data(test_indx,:) * model.Beta;
        weights = model.Beta;
    case 'SVM'
        if size(data,2) > size(data,1)
            data_kernel = data * data';
            [prediction,~,model] = svmconduct(data_kernel, target, ...
                train_indx, test_indx, 'matrix', 1);
            SVs = data(train_indx(model.sv_indices),:);
        else
            [prediction,~,model] = svmconduct( data, target, ...
                train_indx, test_indx, 'linear', 1);
            SVs = model.SVs;
        end
        weights = abs(model.sv_coef'*SVs);
    case 'RF'
        mtry = round(sqrt(size(data,2)));
        t = templateTree('NumPredictorsToSample',mtry,...
            'PredictorSelection','interaction-curvature','Surrogate','on');
        model = fitcensemble(data(train_indx,:),target(train_indx),...
            'Method','bag','NumLearningCycles',500,'FResample',0.63,...
            'Learners',t);
        [~,pred] = predict(model,data(test_indx,:));
        weights = predictorImportance(model);
        prediction = pred(:,2);
    otherwise
        error('Classifier not supported. Try SVM or RF.');
end
perf = fauc(target(test_indx), prediction);



