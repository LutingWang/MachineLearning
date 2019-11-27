clear; clc; close all;
load('data.mat')

%% train
numChanged = 0;
examineAll = true;
while (numChanged > 0 || examineAll)
    numChanged = 0;
    if examineAll
        for i = 1 : m
            numChanged = numChanged + examineExample(i);
        end
    else
        for i = 1 : m
            if 0 < alpha(i) && alpha(i) < C
                numChanged = numChanged + examineExample(i);
            end
        end
    end
    if examineAll
        examineAll = false;
    elseif numChanged == 0
        examineAll = 1;
    end
end
clear numChanged examineAll i

%% visualization
% original data
d0 = data(target == -1, :);
plot(d0(:, 1), d0(:, 2), 'x');
hold on;
d1 = data(target == 1, :);
plot(d1(:, 1), d1(:, 2), '+');

% result
%   note that this result is valid only when kernel takes the form
%   of inner product
w = data' * (alpha .* target);
x = [min(data(:, 1)), max(data(:, 1))];
y = -w(1) / w(2) * x - b / w(2);
plot(x, y);

% support vectors
d = data(alpha > tol, :);
plot(d(:, 1), d(:, 2), 'o');

clear d0 d1 d w x y