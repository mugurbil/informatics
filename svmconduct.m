function [pred_cont,pred_disc,model] = svmconduct(data,target,train_indx,test_indx,kernel,c,dg,cache)
% Conduct the training and prediction of svm classification. It will
% automatically switch to multiclass classification if there is more than
% two classes.
%
%                     . o o O O
%                   _'  ___   _______   _______   _______
%                 _| |_|[] | | [] [] | | [] [] | | [] [] |
%                (   svm   | |       | |       | |       |
%_ _ _ _ _ _ _ _/__o_o_o_o_|=|_o_o_o_|=|_o_o_o_|=|_o_o_o_|=_ _ _ _ _ _ _ _
%-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'-'
%
% INPUTS
% [1] DATA : row major numeric data matrix, each row is a different sample
% [2][a] TARGET : vector of binary target class values. Any binary input
%     automaticaly converted into -1s and 1s. Higher values are assumed to
%     be 1s. If target consists of only one value, it is left as is.
% [2][b] TARGET_INDX : index of target variable if contained in DATA.
% [3] TRAIN_INDX : training indices
% [4] TEST_INDX : testing indices
% [5] KERNEL : svm kernel : linear, poly, poly*, ploy+, rbf, matrix
%     note: 'matrix' is for pre-computed kernel
% [6] C : cost
% [7][a] D : (if kernel == poly[*+]) degree the polynomial
% [7][b] G : (if kernel == rbf) gamma value in kernel, exp(-gamma*|u-v|^2)
% [8] cache : (default 4000) cachesize memory in MB
%
% OUTPUTS
% [1] PRED_CONT : continous prediction values
% [2] PRED_DISC : discrete prediction values
% [3] MODEL : svm model object, all models and labels for multiclass
%
% KERNELS
% [1] LINEAR : linear, data*data'
% [2] POLY : polynomial, (data*data'+1)^D
% [3] POLY* : polynomial, (data*data'+1)^D
% [4] POLY+ : polynomial, (1/size(data,2)*data*data')^D
% [5] RBF : radial basis function,
% [6] MATRIX : pre-computed kernel
% Note: [data*data'](i,j) = data(i,:)*data(j,:)' = u'*v
%   u = x_i = data(i,:)'
%
% LIBSVM OPTIONS
% n-fold cross validation: n must >= 2
% Usage: model = svmtrain(training_label_vector, training_instance_matrix,
%   'libsvm_options');
% libsvm_options:
% -s svm_type : set type of SVM (default 0)
%     0 -- C-SVC
%     1 -- nu-SVC
%     2 -- one-class SVM
%     3 -- epsilon-SVR
%     4 -- nu-SVR
% -t kernel_type : set type of kernel function (default 2)
%     0 -- linear: u'*v
%     1 -- polynomial: (gamma*u'*v + coef0)^degree
%     2 -- radial basis function: exp(-gamma*|u-v|^2)
%     3 -- sigmoid: tanh(gamma*u'*v + coef0)
%     4 -- precomputed kernel (kernel values in training_instance_matrix)
% -d degree : set degree in kernel function
%   (default 3)
% -g gamma : set gamma in kernel function
%   (default 1/num_features)
% -r coef0 : set coef0 in kernel function
%   (default 0)
% -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
%   (default 1)
% -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR
%   (default 0.5)
% -p epsilon : set the epsilon in loss function of epsilon-SVR
%   (default 0.1)
% -m cachesize : set cache memory size in MB
%   (default 100)
% -e epsilon : set tolerance of termination criterion
%   (default 0.001)
% -h shrinking : whether to use the shrinking heuristics, 0 or 1
%   (default 1)
% -b probability_estimates : whether to train a SVC or SVR model for
%   probability estimates, 0 or 1
%   (default 0)
% -wi weight : set the parameter C of class i to weight*C, for C-SVC
%   (default 1)
% -v n : n-fold cross validation mode
% -q : quiet mode (no outputs)

% LOG
% [1.3] (28 Jun 2018) Multiclass classification addition.
% [1.2] (04 Jan 2018) Cache size increase.
% [1.1] (03 Nov 2017) Modern implementation svm conduct.
% [1.0] Based on DSL code.
% 
% Developed by Mehmet Ugurbil

% Copyright (C) 2017-2018 University of Minnesota

if  ~strcmpi(kernel, 'matrix')
    if numel(target) == 1
        variables = setdiff(1:size(data,2),target);
        target = double(data(train_indx,target));
        data_train = double(data(train_indx,variables));
        data_test = double(data(test_indx,variables));
    else
        target = double(target(train_indx));
        data_train = double(data(train_indx,:));
        data_test = double(data(test_indx,:));
    end
    clear data;
end

switch lower(kernel)
    case 'linear'
        options = '-t 0';
    case 'poly'
        options = sprintf('-t 1 -d %d -r 1 -g 1',dg);
    case 'poly*'
        neg = sum(target == -1) / length(target); % Proportion of negatives
        pos = sum(target ==  1) / length(target); % Proportion of positives
        options = sprintf('-t 1 -d %d -r 1 -g 1 -w-1 %g -w1 %g',dg,pos,neg);
    case 'poly+'
        options = sprintf('-t 1 -d %d',dg);
    case 'rbf'
        options = sprintf('-t 2 -g %f',dg);
    case 'matrix'
        options = '-t 4';
        % Compute kernel matrices between every pair of (train,train) and
        % (test,train) instances and include sample serial number as first 
        % column.
        % **To use precomputed kernel, you must include sample serial
        %   number as the first column of the training and testing data.**
        data_train = double([(1:numel(train_indx))',data(train_indx,train_indx)]);
        data_test = double([(1:numel(test_indx))',data(test_indx,train_indx)]);
        clear data;
    otherwise
        except('Unkown kernel.');
end

% Needed objects for training and testing
if nargin < 8; cache = 4000; end
options = sprintf('-q -s 0 -c %f -m %d %s', c, cache, options);
dummy = nan(length(test_indx),1);

classes = unique(target);
% Only one class is needed for binary classification.
if length(classes) == 2
    % Larger value is 1, smaller is -1.
    classes = classes(2);
elseif length(classes) == 1
    % Assume 1 is 1, everything else is -1.
    classes = 1;
end

% Training results
pred_cont = nans(length(test_indx), length(classes));
models = cell(length(classes),1);

% Training
for i = 1:length(classes)
    tmp_target = 2*(target(train_indx) == classes(i)) - 1;
    train_classes = unique(tmp_target);
    % No need for training if only one class.
    if length(train_classes) == 1
        models{i} = [];
        pred_cont(:, i) = train_classes;
        tmp_disc = pred_cont(:, i);
        continue;
    end
    % Modeling
    models{i} = svmtrain(tmp_target, data_train, options);
    [tmp_disc, ~, tmp_pred] = svmpredict(dummy, data_test, models{i}, '-q');
    % Correct if Label(1) ~= 1
    pred_cont(:, i) = tmp_pred * models{i}.Label(1);
end

% Form final prediction - convert back to original class labels
if size(pred_cont,1) > 1
    [~, idx_disc] = max(pred_cont, [], 2);
    pred_disc = classes(idx_disc);
    model = struct('models', models, 'labels', classes);
else
    pred_disc = classes( (tmp_disc == 1) + 1 );
    model = models{1};
end
