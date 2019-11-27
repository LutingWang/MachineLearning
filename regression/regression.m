clear; clc; close all;
load('data.mat');
gradient_descent = false;
kernel = @(x) [x.^2 x ones(length(x), 1)];

%% curve printer
plot(X, y, '+');
hold on;
x = linspace(min(X), max(X), 1000)';
plot_line = @(theta) plot(x, kernel(x) * theta);
clear x;

%% fitting
X = kernel(X);
if (~gradient_descent) % normal equation
    theta = (X' * X) ^ (-1) * X' * y;
    plot_line(theta);
else % gradient descent
    gradient = @(t) X' * (X * t - y);
    theta = zeros(length(X(1,:)), 1);
    alpha = 1e-5;
    tolarence = 20;
    print_counter = 50;
    while (true)
        g = gradient(theta);
        if print_counter == 50
            plot_line(theta);
            print_counter = 1;
        else
            print_counter = print_counter + 1;
        end
        theta = theta - alpha * g;
        if (norm(g) < tolarence)
            break;
        end
    end
    clear gradient alpha tolarence g;
end
clear X y;


