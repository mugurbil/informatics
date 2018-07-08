function [p,r,exitflag] = fisher(T,X,Z,data,handling,threshold)
%FISHER Fisher's exact test.
% 
% INPUTS
% [1] T : target variable index
% [2] X : vector of variables to be tested
% [3] Z : vector of variables to be conditioned on
% [4] data : data matrix, variables as columns
% [5] handling : (default 'complete') how to handle the missing values
%     'complete' use only complete rows, ignore rows with nans
%     'pairwise' use pairs that are not nans
% [6] threshold : (default 5 for 'complete', 10 for 'pairwise')
% Tests whether variables X are independent of a target variable T, given
% the conditioning set Z.
% 
% OUTPUTS
% [1] p :  probability of accepting the null hypothesis H_0: rho(TX|Z) = 0
%     where rho is the partial correlation of T and X given Z.
%     H_0 = T is independent of variables in X, given variables in Z.
%     If p-value  > alpha => Accept H0, i.e. vars are independent (rho == 0)
%     If p-value <= alpha => Reject H0, i.e. vars are dependent   (rho ~= 0)
%     Usually, alpha = 0.05.
% [2] r : Fisher transformation of the partial correlation statistic.
% [3] exitflag : flag indicating reliablity of computation of p-value.
% 
% REFERENCES
% [1] https://en.wikipedia.org/wiki/Partial_correlation
% [2] https://en.wikipedia.org/wiki/Fisher_transformation

% Developed by Mehmet Ugurbil
% Institute for Health Informatics
% University of Minnesota
% 
% April 23rd, 2018
% 
% Based on DSL code.
% The helper functions are copied from "corrcoef" by The MathWorks, Inc.

% Copyright (C) 2018 University of Minnesota

if nargin < 5
    handling = 'complete';
    threshold = 5;
elseif nargin < 6
    if strcmp(handling,'pairwise')
        threshold = 10;
    else
        threshold = 5;
    end
end

const = max(data) - min(data) < eps;
if const(T)
    warning('Target is constant.');
    p = nan;
    r = nan;
    exitflag = 0;
    return;
end

if any(const(Z))
    const = find(const);
    warning('Variable %d is constant. Ignoring.', const);
    Z = setdiff(Z, const);
end

if length(T) > 1 || length(X) > 1
    for i = 1:length(T)
        for j = 1:length(X)
            [p(i,j), r(i,j), exitflag(i,j)] = fisher(T(i), X(j), Z, data, handling, threshold);
        end
    end
    return;
end

% Reindex
data = data(:,[T X Z]);
[N, M] = size(data);
T = 1;
X = 2:(1+length(X));
Z = (2+length(X)):(1+length(X)+length(Z));

switch handling
    case 'complete'
        nans = sum(isnan(data),2) > 0;
        N = N - sum(nans);
        disp(N)
        if N < threshold * M
            warning('Not enough statistical power.');
            p = nan;
            r = nan;
            exitflag = 0;
            return;
        end
        comps = sum(~isnan(data(:,X)) & ~isnan(data(:,T)));
        if comps <= 1
            warning('Not enough statistical power.');
            p = 1;
            r = 0;
            exitflag = 0;
            return;
        end
        S = corrcoef(data,'rows','complete');
    case 'pairwise'
        [S, n] = correlpairwise(data);
        if min(n(:)) < threshold
            warning('Not enough statistical power.');
            p = nan;
            r = nan;
            exitflag = 0;
            return;
        end
end

% S(isnan(S))=0; % correlation coeficient would end up being NaN if one of
% the variables is constant. Here we force it to be zero.

% Compute partial correlations of Z variables using matrix inversion. Then,
% prepare the S matrix to give the recursive formula when plugged in below.
if ~isempty(Z)
    S = S([T X],[T X]) - S([T X],Z) * (S(Z,Z) \ S(Z,[T X]));
end

% Recursive formula to calculate partial correlation
D = diag(S)';
D(D==0) = eps;
r = abs(S(1,2:end))./sqrt(D(1)*D(2:end));

% Fisher's z-transform of the partial correlation and associated p-value
% for two tailed t-test.
z = 0.5 * log((1+r)./(1-r));
df = N - length(Z) - 3;
W = sqrt(df) * z;
p = 2 * tcdf(-abs(W),df);
exitflag = 1;
end

% ------------------------------------------------
function [c, nrNotNaN] = correlpairwise(x)
% apply corrcoef pairwise to columns of x, ignoring NaN entries

n = size(x, 2);

c = zeros(n, 'like', x([])); % using x([]) so that c is always real
nrNotNaN = zeros(n, 'like', x([]));

% First fill in the diagonal:
% Fix up possible round-off problems, while preserving NaN: put exact 1 on the
% diagonal, unless computation results in NaN.
c(1:n+1:end) = sign(localcorrcoef_elementwise(x, x));

% Now compute off-diagonal entries
for j = 2:n
    x1 = repmat(x(:, j), 1, j-1);
    x2 = x(:, 1:j-1);
    
    % make x1, x2 have the same NaN patterns
    x1(isnan(x2)) = nan;
    x2(isnan(x(:, j)), :) = nan;
    
    [c(j,1:j-1), nrNotNaN(j,1:j-1)] = localcorrcoef_elementwise(x1, x2);
end
c = c + tril(c,-1)';
nrNotNaN = nrNotNaN + tril(nrNotNaN,-1)';
nrNotNaN(1:n+1:end) = sum(~isnan(x),1);
% Fix up possible round-off problems: limit off-diag to [-1,1].
t = abs(c) > 1;
c(t) = sign(c(t));

end

% ------------------------------------------------
function [c, nrNotNaN] = localcorrcoef_elementwise(x,y)
%LOCALCORRCOEF Return c(i) = corrcoef of x(:, i) and y(:, i), for all i
% with no error checking and assuming NaNs are removed
% returns 1xn vector c
% x, y must be of the same size, with identical NaN patterns

nrNotNaN = sum(~isnan(x), 1);
xc = x - (sum(x, 1, 'omitnan') ./ nrNotNaN); 
yc = y - (sum(y, 1, 'omitnan') ./ nrNotNaN);

denom = nrNotNaN - 1;
denom(nrNotNaN == 1) = 1;
denom(nrNotNaN == 0) = 0;

xy = conj(xc) .* yc;
cxy = sum(xy, 1, 'omitnan') ./ denom;
xx = conj(xc) .* xc;
cxx = sum(xx, 1, 'omitnan') ./ denom;
yy = conj(yc) .* yc;
cyy = sum(yy, 1, 'omitnan') ./ denom;

c = cxy ./ sqrt(cxx) ./ sqrt(cyy);

% Don't omit NaNs caused by computation (not missing data)
ind = any((isnan(xy) | isnan(xx) | isnan(yy)) & ~isnan(x), 1);
c(ind) = nan;

end