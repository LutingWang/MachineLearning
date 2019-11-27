classdef (Sealed) Output < Layer
    %OUTPUT An output layer in neural network.
    %   Normalize the output of a neural network.
    
    properties (SetAccess = private)
        inputSize;
    end
    
    methods
        function obj = Output(inputSize)
            %OUTPUT Construct an output layer.
            %   Construct an output layer according to the output size of a
            %   neural network.
            obj = obj@Layer(false);
            obj.inputSize = inputSize;
        end
        
        function output = forwardProp(obj, input)
            %FORWARDPROP Normalize the input.
            %   Normalize the input.
            assert(size(input, 1) == obj.inputSize);
            assert(all(all(input >= 0)));
            obj.input = input;
            output = input ./ repmat(sum(input), obj.inputSize, 1);
        end
        
        function delta = backProp(obj, labels)
            %BACKPROP Calculate the output error according to the cost
            %function.
            %   Calculate the output error of cross-entropy cost function.
            delta = obj.input - labels;
        end
    end
end

