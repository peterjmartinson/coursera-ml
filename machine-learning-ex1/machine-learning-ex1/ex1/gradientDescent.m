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

    h = X * theta;
    % fprintf('m: %d, alpha: %d, h: %d\n\n', m, alpha, h);
    % fprintf('J: %d\n', computeCost(X, y, theta));
    % pause;
    temp_theta_1 = theta(1) - alpha * (1/m) * sum( X(:,1)' * (h - y) );
    temp_theta_2 = theta(2) - alpha * (1/m) * sum( X(:,2)' * (h - y) );
    theta = [temp_theta_1; temp_theta_2];

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
