function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));		% returns vector [nr, nc] where 'nr' = number rows, 'nc' = number columns
						% note: g is now a matrix of zeros w rows / columns equal to z


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


%	S(t)= (1 / (1 + e^(-z)))

g = (1 ./ (1 + exp(-z)));

% =============================================================

end
