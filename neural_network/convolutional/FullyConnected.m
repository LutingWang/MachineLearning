classdef (Sealed) FullyConnected < ComputationalLayer
    %FULLYCONNECTED A fully-connected layer in neural networks.
    %   A fully-connected layer in neural networks.
    
    properties (SetAccess = private)
        inputSize;
        outputSize;
    end
    
    methods (Access = protected)
        function initWeights(obj)
            %INITWEIGHTS Initialize weights randomly.
            %   Initialize weights randomly.
            obj.weights = Layer.normalizeWeights( ...
                rand(obj.outputSize, obj.inputSize + 1) ...
                );
        end
    end
    
    methods
        function obj = FullyConnected(activated, inputSize, outputSize)
            %FULLYCONNECTED Construct a fully-connected layer.
            %   Construct a fully-connected layer which maps from
            %   `inputSize` (not counting bias) to `outputSize`.
            obj = obj@ComputationalLayer(activated);
            obj.inputSize = inputSize;
            obj.outputSize = outputSize;
            obj.initWeights();
        end
        
        function output = forwardProp(obj, input)
            %FORWARDPROP Perform a forward propagation.
            %   Perform a forward propagation without recording the
            %   `input`. The `input` should be (`inputSize` * m) where m
            %   stands for batch size. After pre-processing, the stored
            %   `obj.input` would be ((`inputSize` + 1) * m)
            assert(size(input, 1) == obj.inputSize);
            m = size(input, 2);
            obj.input = [ones(1, m); input];
            output = Layer.ACTIVATOR(obj.weights * obj.input);
        end
        
        function delta_prime = backProp(obj, delta)
            %BACKPROP Perform a back propagation.
            %   Perform a back propagation according to `delta` passed in
            %   from the following layer.
            assert(size(delta, 1) == obj.outputSize);
            m = size(obj.input, 2);
            assert(size(delta, 2) == m);
            delta_prime = obj.weights' * delta ...
                .* obj.f(obj.input);
            delta_prime(1, :) = [];
            assert(size(delta_prime, 1) == obj.inputSize);
            gradient = (delta * obj.input') / m + Layer.LAMBDA * obj.weights;
            obj.weights = obj.weights - Layer.ALPHA * gradient;
        end
    end
end

