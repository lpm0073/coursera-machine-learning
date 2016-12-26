function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Regularized Cost Function in the form
% J(theta) = (1 / (2*m)) * sum((h(x) - y)^2) + (lambda / (2 * m)) * sum(theta^2)
hX = X * theta;					% NOTE: we'll re-use this term for the gradient
J 	= (1 / (2 * m)) * sumsq(hX - y) + (lambda / (2 * m)) * sumsq(theta(2:end));

% Regularized Gradient Function in the form
% d(theta, j) = (1 / m) * sum(((theta' * x) - y) * x_j) 							% for j = 0
% d(theta, j) = (1 / m) * sum(((theta' * x) - y) * x_j) + (lamda / m) * theta_j		% for j >= 0

grad 		= (1 / m) * (X' * (hX - y));
grad(2:end) +=  (1 / m) * lambda * theta(2:end);

% =========================================================================

%grad = grad(:);				% this shouldn't be necesary

end
