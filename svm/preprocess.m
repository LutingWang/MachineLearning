% parameters
tol = 1e-3;
C = 1;
k = @(v1, v2) v1 * v2';

% load data
data = csvread('iris.data');
data = data(:, [3, 4, 5]);
data = data(logical(data(:, 3)), :);
data = [data; 7, 1.5, 1]; % include error point
target = 2 * logical(data(:, 3) - 1) - 1;
data(:, 3) = [];

% data parameters
m = length(target);
n = 2;

% setup kernel and eta matrix
kernel = zeros(m, m);
eta = zeros(m, m);
for i = 1 : m
    kernel(i, i) = k(data(i, :), data(i, :));
    eta(i, :) = eta(i, :) + kernel(i, i);
    eta(:, i) = eta(:, i) + kernel(i, i);
end
for i = 1 : m
    for j = 1 : i - 1
        kernel(i, j) = k(data(i, :), data(j, :));
        kernel(j, i) = kernel(i, j);
    end
end
eta = eta - 2 * kernel;

% initialize variables
alpha = zeros(m, 1);
b = 0;

% save to data.mat
save data.mat data target m n kernel eta alpha b tol C