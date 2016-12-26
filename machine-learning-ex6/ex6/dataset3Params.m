function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%     Inf:  Return a scalar, matrix or N-dimensional array whose elements are
%     all equal to the IEEE representation for positive infinity.
maxError = Inf;   % <----- thank you google.

% So ...... not very sophisticated, but, nested loop brute force method to try EVERY combination of C and Sigma
% Keep track of which pair produces the smallest 'cost' and then, Bob's your uncle.
%
% Note: the input vectors come directly from page 7 of the exercise instructions (so, no rocket science here either.)
for C_i = [0.01 0.03 0.1 0.3 1 3 10 30]
  for Sigma_j = [0.01 0.03 0.1 0.3 1 3 10 30]

    model   = svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, Sigma_j)); 
    y_ij    = svmPredict(model, Xval);
    J_ij    = mean(double(y_ij ~= yval));   % Note: see page 9 footnote, "Implementation Tip"

    if J_ij < maxError
      fprintf(['Found better values: C = %f and sigma = %f \n'], C, sigma);

      maxError  = J_ij;
      C         = C_i;
      sigma     = Sigma_j;
    end
  end
end

fprintf(['All done! final values: C = %f and sigma = %f \n'], C, sigma);


% =========================================================================

end
