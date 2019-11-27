function updated = examineExample(i2)
    updated = 0;

    % check whether alpha2 violated the KKT conditions
    target = evalin('base', 'target');
    alpha = evalin('base', 'alpha');
    tol = evalin('base', 'tol');
    C = evalin('base', 'C');
    r2 = E(i2) * target(i2);
    if ~(r2 < -tol && alpha(i2) < C) && ~(r2 > tol && alpha(i2) > 0)
        return;
    end
    
    % select i1
    eta = evalin('base', 'eta');
    nonBound = 0 < alpha & alpha < C;
    if sum(nonBound) ~= 0
        [~, i1] = max(abs(E() - E(i2)) .* nonBound ./ eta(:, i2));
        if takeStep(i1, i2)
            updated = 1;
            return;
        end
    end
    
    m = evalin('base', 'm');
    randstart = randi([1, m], [1, 1]);
    for i = randstart : randstart + m - 1
        if i > m
            i = i - m;
        end
        if nonBound(i) && takeStep(i, i2)
            updated = 1;
            return;
        end
    end
    for i = randstart : randstart + m - 1
        if i > m
            i = i - m;
        end
        if ~nonBound(i) && takeStep(i, i2)
            updated = 1;
            return;
        end
    end
    
    
