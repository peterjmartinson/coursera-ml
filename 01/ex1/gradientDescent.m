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
    sizem = size(m);
    sizeX = size(X);
    sizey = size(y);
    sizet = size(theta);
    sizep = size(predictions);
    fprintf('m: %d,%d\n', sizem[1], sizem[2]);
    fprintf('X: %d,%d\n', sizeX[1], sizeX[2]);
    fprintf('y: %d,%d\n', sizey[1], sizey[2]);
    fprintf('theta: %d,%d\n', sizet[1], sizet[2]);
    fprintf('predictions: %d,%d\n', sizep[1], sizep[2]);
    sqrErrors = (predictions-y)*X;
    temp = 1/m * sum(sqrErrors);
    theta = temp;







    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
