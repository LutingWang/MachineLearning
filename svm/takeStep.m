function updated = takeStep(i1, i2)
    updated = 0;

    % check for validity
    eta = evalin('base', 'eta');
    eta = eta(i1, i2);
    if eta <= 0 % indicating i1 == i2 in this case
        return;
    end

    % calculate H and L
    target = evalin('base', 'target');
    C = evalin('base', 'C');
    alpha = evalin('base', 'alpha');
    if target(i1) == target(i2)
        H = min(C, alpha(i2) + alpha(i1));
        L = max(0, alpha(i2) + alpha(i1) - C);
    else
        H = min(C, C + alpha(i2) - alpha(i1));
        L = max(0, alpha(i2) - alpha(i1));
    end
    if (L >= H)
        return;
    end
    
    % calculate alpha
    tol = evalin('base', 'tol');
    a2 = alpha(i2) + target(i2) * (E(i1) - E(i2)) / eta;
    if a2 > H
        a2 = H;
    elseif a2 < L
        a2 = L;
    end
    if abs(a2 - alpha(i2)) < tol * (a2 + alpha(i2) + tol)
        return;
    end
    a1 = alpha(i1) + target(i1) * target(i2) * (alpha(i2) - a2);
    
    % update threshold
    kernel = evalin('base', 'kernel');
    b = evalin('base', 'b');
    b1 = b - E(i1) - target(i1) * (a1 - alpha(i1)) * kernel(i1, i1) ...
        - target(i2) * (a2 - alpha(i2)) * kernel(i1, i2);
    b2 = b - E(i2) - target(i1) * (a1 - alpha(i1)) * kernel(i2, i1) ...
        - target(i2) * (a2 - alpha(i2)) * kernel(i2, i2);
    if 0 < a1 && a1 < C
        b = b1;
    elseif 0 < a2 && a2 < C
        b = b2;
    else
        b = (b1 + b2) / 2;
    end
    
    % store data
    alpha(i1) = a1;
    alpha(i2) = a2;
    assignin('base', 'b', b);
    assignin('base', 'alpha', alpha);
    
    updated = 1;
end