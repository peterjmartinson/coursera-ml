function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
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

    m = size(X,1);
    predictions = X*theta;
    sqrErrors_1 = (predictions-y)*X(1);
    sqrErrors_2 = (predictions-y)*X(2);
    temp_1 = theta(1) - alpha * (1/m * sum(sqrErrors_1));
    temp_2 = theta(2) - alpha * (1/m * sum(sqrErrors_2));
    theta(1) = temp_1;
    theta(2) = temp_2;
    fprintf('theta: (%d, %d)\n', theta(1), theta(2));
    fprintf('J: %d\n', computeCost(X, y, theta));

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
