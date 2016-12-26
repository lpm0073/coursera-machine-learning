function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    % Theta(j) = (alpha * (1 / m)) * sum (error(i)) * x(i)j
    % so ......

    x = X(:, 2);                                                             % all learning sets of X, 2nd column (the "X's")
    hypothesis = theta(1) + (theta(2) * x);                                  % hypothesis in form of y = B + mx

    thetaB      = theta(1) - alpha * (1 / m) * sum(hypothesis - y);          % y intercept
    thetaMX     = theta(2) - alpha * (1 / m) * sum((hypothesis - y) .* x);   % slope (mX)

    theta = [thetaB; thetaMX];


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
