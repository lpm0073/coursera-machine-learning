function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

% Note:	1. this is part 2 of 2 for mapping a linear transformation from N-space to K-space
%		2. X contains pixel values from an uncompressed database of facial images.

% Note: for cookbook recipe see lecture 14 notes, page 19.
% From an earlier step in this same exercise we created pca.m that, combined with ex7_pca takes care of:
%	a.) feature-normalize and feature-scale the data. 
%		the course organizers already passed X thru featureNormalize() in the calling ex7_pca.m
%	b.) calculate the covariance matrix Sigma = (1/m) * X' * X
%	c.) call Octave built-in fuction [U,S,V] = svd(Sigma);
%
%	here, we take care of the rest of the exercise  ....
%	* d.) take the first 4 rows from the reduction matrix Ureduce = U(:,1:k)
%	* e.) after much linear algegra brain damage, it can be shown that z = Ureduce’*x;
%
% Note:
% U = eigenvectors of X
% S = eigenvalues (identity matrix style) -- per lecture video: sum the first k columns of this matrix to get (?????) something important. WTF?????


U_reduce = U(:, 1:K);					% U contains eigen values stored in the form of an identity matrix

for i = 1: size(X, 1)
	Z(i, :) = (U_reduce' * X(i, :)')';	% 1 row per image pixel, 1 column per K (dimension reduction quantity)
end
					% Note: in more plain english, this is producing the slope of the vector Z from its original 
					%		N-space form. 


% =============================================================

end
