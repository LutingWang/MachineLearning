function val = E(varargin)
    narginchk(0, 1);
    kernel = evalin('base', 'kernel');
    target = evalin('base', 'target');
    alpha = evalin('base', 'alpha');
    b = evalin('base', 'b');
    val = kernel * (alpha .* target) + b - target;
    if nargin == 1
        val = val(varargin{1});
    end
end