data = xlsread('AirQualityUCI.xlsx');
data = data(:, [4, 6]);
data(data(:, 1) == -200 | data(:, 2) == -200, :) = [];
data = mapstd(data')';
X = data(:, 2);
y = data(:, 1);
save data.mat X y