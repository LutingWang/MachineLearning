classdef ComputationalLayer < Layer
    %COMPUTATIONALLAYER Template for a computational layer in neural
    %network.
    %   A computational layer stores its weights and does computations in
    %   forward and back propagations.
    
    properties (SetAccess = protected)
        weights;
    end
    
    methods (Access = protected)
        function obj = ComputationalLayer(activated)
            %COMPUTATIONALLAYER Construct a computational layer.
            %   Construct a computational layer.
            obj = obj@Layer(activated);
        end
    end

    methods (Abstract, Access = protected)
        initWeights(obj);
    end
end

