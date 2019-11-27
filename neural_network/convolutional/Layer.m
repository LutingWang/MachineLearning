classdef Layer < handle
    %LAYER Template for a layer in neural networks.
    %   Base class of full-connected layers, convolutional layers, and
    %   pooling layers, etc.
    
    properties (Constant)
        EPSILON = 1;
        LAMBDA = 0.01;
        ALPHA = 0.05;
        ACTIVATOR = @(z) 1 ./ (1 + exp(-z));
        RELU = @(z) max(z, 0);
    end
    
    properties
        input;
        f = @(a) a .* (1 - a);
    end
    
    methods (Static)
        function normalized = normalizeWeights(original)
            %NORMALIZEWEIGHTS Helps normalize randomly initialized weights.
            %   Constraint the weights within (-epsilon, epsilon), suppoing
            %   that the weights are drawn from Unif(0, 1).
            normalized = original * 2 * Layer.EPSILON - Layer.EPSILON;
        end
    end
    
    methods (Access = protected)
        function obj = Layer(activated)
            %LAYER Construct a layer.
            %   If the former layer is activated, hold `f` as its default
            %   value (derivative of sigmoid). Otherwise, set `f` to a
            %   constant of 1.
            if ~activated
                obj.f = @(a) 1;
            end
        end
    end
    
    methods (Abstract)
        output = forwardProp(obj, input);
        delta_prime = backProp(obj, delta);
    end
end

