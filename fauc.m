function varargout = fauc(target, pred, pred_2)
%FAUC Fast Area Under Curve calculator and comparer. Computes the area
% under curve (AUC) for the reciever operating characteristic (ROC) for
% binary classification.
%
%   A = FAUC(TARGET, PRED) returns the AUC for the prediction vector PRED
%   and the vector TARGET. TARGET is a binary target variable vector.
%   Higher of the two class values is assumed to be of the positive class.
%   PRED is classification prediction vector that can be sorted. Higher
%   prediction values are assumed to indicate higher degree of confidence
%   that an instance belongs to the positive class.
%
%   [A, P, CI_L, CI_U, SE_A] = FAUC(TARGET, PRED) returns the 95%
%   confidence interval of the AUC. P is the p value of the confidence
%   interval, CI_L the lower bound, CI_U the upper bound, and SE_A the
%   standard error of the AUC. The p value of the confidence interval is
%   the probability that given an AUC value in this confidence interval,
%   it is statistically indistinguishable from current AUC.
%
%   [P, Q, A_1, A_2] = FAUC(TARGET, PRED_1, PRED_2) compares the two
%   predictions and returns the p value P, comparator Q as well as the
%   two AUC values for the given predictions A_1 and A_2 respectively.
% 
% COMMENTS
% Mann-Whitney U Statistics is used to calculate the area.
%
% X = positive elements (Class 1)
% Y = negative elements (Class 0)
%
% n_X = number of positive elements
% n_Y = number of negative elements
%
% P = Psi(X,Y) = { 1   if Y < X
%                { 1/2 if Y = X
%                { 0   if Y > X
%
% V_X = row sums of psi
%     = V(X) = [V(X_i)]{i=1:m}
%               V(X_i) = 1/n SUM{j=1:n} Psi(X_i,Y_j)
% I_X = number of inversions in rows
%     = (1-V_X)*n_X
% V_Y = column sums of psi
%     = V(Y) = [V(Y_j)]{j=1:n}
%               V(Y_j) = 1/m SUM{i=1:m} Psi(X_i,Y_j)
% I_Y = number of inversions in columns
%     = (1-V_Y)*n_Y
%
% A   = AUC for the given prediction
% S   = variance of the AUC
% C   = covariance of the AUC
%
% P : p value for the null hypothesis that predictions are similar
%     p < threshold => predictions are different
%     p > threshold => predictions have similar AUCs
%
% Q : comparative  {-1 if AUC_1 < AUC_2
%                  { 0 if AUC_1 = AUC_2
%                  {+1 if AUC_1 > AUC_2
%
%        Reciever Operating Characteristic Curve
%             _____________________________.
%          1 |                  ______/ .  |
%            |              ___/     .     |
%            |          ___/      .        |
%            |       __/       .           |
%     True   |     _/       .              |
%   Positive |    /      .                 |
%     Rate   |  _|    .                    |
%            | /   .                       |
%            |/ .                          |
%            ._____________________________|
%           0      False Positive Rate     1
% 
% REFERENCES
% [1] DeLong, E.R., DeLong, D.M. and Clarke-Pearson D.L. (1988). "Comparing
% the Areas under Two or More Correlated Receiver Operating Characteristic
% Curves: A Nonparametric Approach." Biometrics vol 44, no 3,  pp. 837-845.
% (https://www.jstor.org/stable/2531595)

% LOG
% [1.3] (15 May 2018) Bug fix. Remove case when length(u_pred) = 1 (calc).
%       Correct p-value in calc if se_a = 0 to 1.
% [1.2] (18 Mar 2018) Added normalcdf funtion in case normcdf is not
%       installed (Statistics Toolbox).
% [1.1] (8 Jan 2018) Bug fix. c_x (or c_y) is not 2x2 if i_x_2 (or i_y_2)
%       is 1 or 0. Need to set c_x (or c_y) to zeros(2).
% [1.0] (2 Jan 2018) Developed in MATLAB 2018a.
% 
% Developed by Mehmet Ugurbil

% Copyright (C) 2018 University of Minnesota

target = target(:);
pred = pred(:);

labels = unique(target);
if length(labels) < 2
    varargout = {nan, nan, nan, nan, nan};
    return;
end

% assert(length(labels) == 2, 'Target (1) must be binary.');
target = (target==labels(2));

assert(length(target) == length(pred), ...
    'Length of prediction (2) must match length of target (1).');

if nargin == 2
    idx = ~isnan(pred);
    pred = pred(idx);
    target = target(idx);
    if nargout == 1
        varargout{1} = calc(target, pred);
    else
        [A, ~, ~, p, ci_l, ci_u, se_a] = calc(target, pred);
        varargout = {A, p, ci_l, ci_u, se_a};
    end
elseif nargin == 3
    pred_2 = pred_2(:);
    assert(length(target) == length(pred_2), ...
        'Length of prediction (3) must match length of target (1).');
    idx = ~isnan(pred) & ~isnan(pred_2);
    pred = pred(idx);
    target = target(idx);
    pred_2 = pred_2(idx);
    [p, q, a1, a2] = compare(target, pred, pred_2);
    varargout = {p, q, a1, a2};
end

% -------------------- %
function [p, q, a_1, a_2] = compare(target, pred_1, pred_2)
% Compare two predictions.

[a_1, i_x_1, i_y_1] = calc(target, pred_1);
[a_2, i_x_2, i_y_2] = calc(target, pred_2);

if a_1 == a_2
    p = 1; q = 0;
    return;
end

n_x = sum( target);
n_y = sum(~target);

c_x = cov(i_x_1, i_x_2);
if numel(c_x) == 1
    c_x = zeros(2);
end

c_y = cov(i_y_1, i_y_2);
if numel(c_y) == 1
    c_y = zeros(2);
end

c = c_x(1,2)*n_x + c_y(1,2)*n_y;
s_1 = c_x(1,1)*n_x + c_y(1,1)*n_y;
s_2 = c_x(2,2)*n_x + c_y(2,2)*n_y;

s = sqrt(s_1 + s_2 - 2*c)/(n_x*n_y);
p = 2 * (1 - normalcdf(abs(a_2-a_1)/s));
q = (a_1 > a_2) - (a_1 < a_2);

% -------------------- %
function [a, i_x, i_y, p, ci_l, ci_u, se_a] = calc(target, pred)
% Calculate the AUC and the related variables.

[u_pred, indx, ~] = unique(pred);

n_x = sum(target);
n_y = sum(~target);

if length(u_pred) == length(pred)
    % All predictions are unique
    target = target(indx);
    ranks_x = find(target);
    ranks_y = find(~target);
    if nargout > 1
        sub_ranks_x = (1:n_x)';
        sub_ranks_y = (1:n_y)';
    end
else
    % There are ties
    [pred, indx] = sort(pred);
    target = target(indx);
    ranks = ranking(pred);
    ranks_x = ranks(target);
    ranks_y = ranks(~target);
    if nargout > 1
        sub_ranks_x = ranking(pred(target));
        sub_ranks_y = ranking(pred(~target));
    end
end

sum_R = sum(ranks_x);

UStat = sum_R - n_x*(n_x+1)/2;
a = UStat/(n_y*n_x);

if nargout > 1
	i_x = n_y - ranks_x + sub_ranks_x;
	i_y = ranks_y - sub_ranks_y;
end

% Statistics for the confidence interval
if nargout > 3
    perc = 1.96; % Approximate value of the 97.5 percentile of N(0,1)
    se_a = sqrt(var(i_x)*n_x + var(i_y)*n_y)/(n_x*n_y);
	p = (1-normalcdf(abs((a-0.5)/se_a)));
    if ~se_a; p = 1; end
	ci_l = max(a - perc*se_a,0);
	ci_u = min(a + perc*se_a,1);
end

% -------------------- %
function ranks = ranking(pred)
% Computes the ranks when ties are present. Ties are ranked the same value
% as the average of their indices. Use "find" when no ties for speed.
%   ** Pred needs to be a sorted array **

ranks = zeros(length(pred),1);
last = pred(1);
i0 = 1;
for i = 2:length(pred)
    if pred(i) > last
        ranks(i0:(i-1)) = (i0+i-1)/2;
        i0 = i;
        last = pred(i);
    end
end
ranks(i0:i) = (i0+i)/2;

% -------------------- % 
function f = normalcdf(z)
% In case normcdf function is not installed.

try
    f = normcdf(z);
catch
    f = 0.5 * erfc(-z ./ sqrt(2));
end
