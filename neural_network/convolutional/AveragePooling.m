classdef (Sealed) AveragePooling < Layer
    %AVERAGEPOOLING An average pooling layer in neural networks.
    %   An average pooling layer in neural networks.
    
    properties (SetAccess = protected)
        blockSize;
    end
    
    methods
        function obj = AveragePooling(activated, blockSize)
            %AVERAGEPOOLING Construct an average pooling layer.
            %   Construct a average pooling layer which, for each layer of
            %   input, takes the average of each block as the
            %   representation of the block.
            obj = obj@Layer(activated);
            obj.blockSize = blockSize;
        end
        
        function output = forwardProp(obj, input)
            %FORWARDPROP Detect the size of `input` and perform a forward
            %propagation.
            %   Detect the size of `input` and perform a forward
            %   propagation.
            assert(ndims(input) == 4);
            m = size(input, 4);
            c = size(input, 3);
            assert(size(input, 1) == size(input, 2));
            s = size(input, 1);
            assert(mod(s, obj.blockSize) == 0);
            obj.input = input;
            
            output = blockproc(reshape(input, s, s * c, m), ...
                [obj.blockSize, obj.blockSize], ...
                @(block) mean(mean(block.data)) ...
                );
            output = reshape(output, s / obj.blockSize, ...
                s / obj.blockSize, c, m);
        end
        
        function delta_prime = backProp(obj, delta)
            %BACKPROP Assign the error evenly to each input fileds.
            %   Assign the error evenly to each input fileds.
            delta_prime = repelem(delta / (obj.blockSize ^ 2), ...
                obj.blockSize, obj.blockSize) .* obj.f(obj.input);
        end
    end
end
