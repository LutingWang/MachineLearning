classdef (Sealed) Flatten < Layer
    %FLATTEN An flatten layer in neural network.
    %   Flatten the input images to one dimensional vectors.
    
    properties (SetAccess = protected)
        inputSize;
        inputLayers;
    end
    
    methods
        function obj = Flatten(activated, inputSize, inputLayers)
            %Flatten Construct a flatten layer.
            %   Construct a flatten layer used to flatten images with its
            %   size being (`inputSize` * `inputSize` * `inputLayers`).
            obj = obj@Layer(activated);
            obj.inputSize = inputSize;
            obj.inputLayers = inputLayers;
        end
        
        function output = forwardProp(obj, input)
            %FORWARDPROP Flatten input.
            %   Flatten input.
            assert(ndims(input) == 4);
            assert(size(input, 1) == obj.inputSize);
            assert(size(input, 2) == obj.inputSize);
            assert(size(input, 3) == obj.inputLayers);
            obj.input = input;
            
            output = reshape(input, obj.inputSize ^ 2 * obj.inputLayers, size(input, 4));
        end
        
        function delta_prime = backProp(obj, delta)
            %BACKPROP Reshape the error.
            %   Reshape the error.
            assert(ismatrix(delta));
            delta_prime = reshape(delta, obj.inputSize, obj.inputSize, ...
                obj.inputLayers, size(delta, 2)) .* obj.f(obj.input);
        end
    end
end

