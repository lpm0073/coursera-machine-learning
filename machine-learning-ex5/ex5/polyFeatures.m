function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%


% You need to return the following variables correctly.
X_poly = zeros(numel(X), p);

% ====================== YOUR CODE HERE ======================
% Instructions: Given a vector X, return a matrix X_poly where the p-th 
%               column of X contains the values of X to the p-th power.
%
% 



% Note: this snippet works solely by accident. the smart piece of the strategy is vectorizing
% 		the iteration 1:m. however, we TECHNICALLY shouldn't manipulate the first order vector (p=1)
%		be that as it may, X^p = X when p=1, so, no harm done :)
for i=1:p
	X_poly(:,i) = X.^i;
end

% Note: the following, while syntactically more tedious is technically more accurate.
%X_poly(:,1) = X;
%for i=2:p
%    X_poly(:,i) = X.*X_poly(:,i-1);
%end


% =========================================================================

end
