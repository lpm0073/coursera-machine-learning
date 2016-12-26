function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix. 
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

% Note: for cookbook recipe see lecture 14 notes, page 19.
%	a.) feature-normalize and feature-scale the data. 
%		the course organizers already passed X thru featureNormalize() in the calling ex7_pca.m
%	b.) calculate the covariance matrix Sigma = (1/m) * X' * X
%	c.) call Octave built-in fuction [U,S,V] = svd(Sigma);
%
%	then, in the next part of the exercise  ....
%	* d.) take the first 4 rows from the reduction matrix Ureduce = U(:,1:k)
%	* e.) after much linear algegra brain damage, it can be shown that z = Ureduce’*x;
%
% Note:
% U = eigenvectors of X
% S = eigenvalues (identity matrix style) -- per lecture video: sum the first k columns of this matrix to get (?????) something important. WTF?????

Sigma 		= X' * X/ m;
[U,S,V] 	= svd(Sigma);				% yes, these two lines really do everything. All hail Octave!


% =========================================================================

end
