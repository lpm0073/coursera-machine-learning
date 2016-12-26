function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

% Basic Guassian (Normal) Distribution inputs defined as.....
% 	mu = (1 / m) * SUM(Xi)
% 	sigma^2 = (1 / m) * SUM (Xi - mu)^2

%	Built-in Function:  bsxfun (F, A, B)
%     The binary singleton expansion function applier performs
%     broadcasting, that is, applies a binary function F
%     element-by-element to two array arguments A and B, and expands as
%     necessary singleton dimensions in either input argument.  F is a
%     function handle, inline function, or string containing the name of
%     the function to evaluate.  The function F must be capable of
%     accepting two column-vector arguments of equal length, or one
%     column vector argument and a scalar.

%     The dimensions of A and B must be equal or singleton.  The
%     singleton dimensions of the arrays will be expanded to the same
%     dimensionality as the other array.


mu 		= mean(X);
sigma2	= (1 / m) * sum((bsxfun(@minus, X, mu)).^2);	% <--- Looks weird, but, this is exactly (1 / m) * SUM (Xi - mu)^2


%Alternate Implementation I: ---------------------------------------------------------------
%	mu = 1/m .* sum(X);
%	diff = mu' * ones(1, m);
%	diff = diff';
%	sigma2 = 1/m .* sum((X - diff).^2);

%Alternate Implementation II: ---------------------------------------------------------------

%	mu=(mean(X))'; %mu must be a column vector but features are in columns, so I have to transpose it
%	mu_rep=(repmat(mu,1,m))'; %replicate mu to have the same size of X (now it is m x n because I transpose the replicated matrix)
%	X_sub=X-mu_rep; %subtract means
%	sigma2=((1/m)*sum(X_sub.^2))';


% =============================================================


end
