function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% In theory this line shouldnt be necesary. of course, in theory, the input parameter 
% shouldnt have been necesary since it can be derived.
% num_labels = size(Theta2, 1);
%
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------
% Part 1: Feedforward the neural network and return the cost in the variable J

a1 	= [ones(1, m); X'];  					% (Bias unit + 400 input dimensions) x 5,000 (training sets)
z2 	= Theta1 * a1;							% <-- NOTE: NO MATRIX INVERSION becaues Theta1 = 25x401 | a1 = 401 x 5,000
a2 	= [ones(1, m); sigmoid(z2)];  			% 26 x 5,000. Note sigmoid function was provided by course 
a3 	= sigmoid(Theta2 * a2);  				% a3 = 10 classifications x 5,000 training sets. Theta2 = 10x26 | a2 = 26x5,000

%----------------------------
% reclass y from 1:10 scalar to vectorized identity format
%
% 	  sub2ind: Convert subscripts to a linear index.
%     The following example shows how to convert the two-dimensional
%     index `(2,3)' of a 3-by-3 matrix to a linear index.  The matrix is
%     linearly indexed moving from one column to next, filling up all
%     rows in each column.
%
%          linear_index = sub2ind ([3, 3], 2, 3)
%          => 8
%
%     See also: ind2sub.
%----------------------------
Y 	= zeros(num_labels, m);						% 10x10
Y(sub2ind(size(Y), y', 1:m)) = 1;				% vectorized transform of y to Y. 


J 	= (1 / m) * sum(sum(-Y .* log(a3) - (1 - Y) .* log(1 - a3)));

% Add regularized error. Drop the bias terms in the 1st columns.
J 	= J + (lambda / (2 * m)) * sum(sum(Theta1(:, 2:end) .^ 2));
J 	= J + (lambda / (2 * m)) * sum(sum(Theta2(:, 2:end) .^ 2));


% -------------------------------------------------------------
% Part 2: Implement the backpropagation algorithm to compute the gradients Theta1_grad and Theta2_grad

% See Page 7 of lecture notes. 	d3 is a reduction of form (Theta3' * d4) .* g'(z3)
%								d2 is an implementation of form (Theta2' * d3) .* g'(z2)
% 
d3 	= a3 - Y;  														% 10 x 5,000 (see above)
d2 	= (Theta2' * d3) .* [ones(1, m); sigmoidGradient(z2)];  		% (25 units + Bias) x 5,000

Theta2_grad 	= (1 / m) * d3 * a2';
Theta1_grad 	= (1 / m) * d2(2:end, :) * a1';

% add lambda to regularize. Note: do not regularize the Bias unit (dimension 0)
Theta2_grad = Theta2_grad + (lambda / m) * ([zeros(size(Theta2, 1), 1), Theta2(:, 2:end)]);
Theta1_grad = Theta1_grad + (lambda / m) * ([zeros(size(Theta1, 1), 1), Theta1(:, 2:end)]);



% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
