classdef (Sealed) Convolutional < ComputationalLayer
    %CONVOLUTIONAL A convolutional layer in neural networks.
    %   A convolutional layer in neural networks.
    
    properties (SetAccess = private)
        kernelSize;
        inputLayers;
        outputLayers;
        bias;
    end
    
    methods (Access = protected)
        function initWeights(obj)
            %INITWEIGHTS Initialize weights randomly.
            %   Initialize weights randomly.
            obj.weights = Layer.normalizeWeights( ...
                rand(obj.kernelSize, obj.kernelSize, ...
                obj.inputLayers, obj.outputLayers) ...
                );
            obj.bias = Layer.normalizeWeights(rand(obj.outputLayers, 1));
        end
    end
    
    methods
        function obj = Convolutional(activated, kernelSize, ...
                inputLayers, outputLayers)
            %CONVOLUTIONAL Construct a convolutional layer.
            %   Construct a convolutional layer.
            obj = obj@ComputationalLayer(activated);
            obj.kernelSize = kernelSize;
            obj.inputLayers = inputLayers;
            obj.outputLayers = outputLayers;
            obj.initWeights();
        end
        
        function output = forwardProp(obj, input)
            %FORWARDPROP Perform a forward propagation.
            %   For each output layer, compute the 3D convolution between
            %   `input` and the corresponding kernel.
            assert(ndims(input) == 4);
            m = size(input, 4);
            assert(size(input, 3) == obj.inputLayers);
            assert(size(input, 1) == size(input, 2));
            obj.input = input;
            outputSize = size(input, 1) - obj.kernelSize + 1;
            output = zeros(outputSize, outputSize, obj.outputLayers, m);
            for i = 1 : m
                for c = 1 : obj.outputLayers
                    output(:, :, c, i) = convn(input(:, :, :, i), ...
                        obj.weights(:, :, :, c), 'valid') + obj.bias(c);
                end
            end
            output = Layer.RELU(output);
        end
        
        function delta_prime = backProp(obj, delta)
            %BACKPROP Perform a back propagation.
            %   Perform a back propagation.
            assert(ndims(delta) == 4);
            m = size(delta, 4);
            assert(m == size(obj.input, 4));
            assert(size(delta, 3) == obj.outputLayers);
            assert(size(delta, 1) == size(delta, 2));
            inputSize = size(delta, 1) + obj.kernelSize - 1;
            assert(inputSize == size(obj.input, 1));
            
            delta_prime = zeros(size(obj.input));
            gradient_w = zeros(size(obj.weights));
            gradient_b = zeros(size(obj.bias));
            for i = 1 : m
                for c = 1 : obj.outputLayers
                    delta_layer = delta(:, :, c, i);
                    delta_prime(:, :, :, i) = delta_prime(:, :, :, i) + ...
                        convn(delta_layer, ...
                        rot90(rot90(obj.weights(:, :, :, c)))) .* ...
                        obj.f(obj.input);
                    gradient_w(:, :, :, c) = gradient_w(:, :, :, c) + ...
                        convn(obj.input(:, :, :, i), delta_layer, 'valid');
                    gradient_b(c) = gradient_b(c) + sum(sum(delta_layer));
                end
            end
            delta_prime = delta_prime .* obj.f(obj.input);
            
            gradient_w = gradient_w / m + Layer.LAMBDA * obj.weights;
            gradient_b = gradient_b / m + Layer.LAMBDA * obj.bias;
            obj.weights = obj.weights - Layer.ALPHA * gradient_w;
            obj.bias = obj.bias - Layer.ALPHA * gradient_b;
        end
    end
end

