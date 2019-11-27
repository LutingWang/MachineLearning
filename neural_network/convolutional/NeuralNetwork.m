classdef NeuralNetwork < handle
    %NEURALNETWORK Template for a neural network.
    %   A neural network that consists of layers.
    
    properties
        layers;
        outputLayer;
    end
    
    methods (Static)
        function accuracy = evaluate(prediction, label)
            [~, predLab] = max(prediction);
            accuracy = 0;
            for i = 1 : size(label, 2)
                accuracy = accuracy + label(predLab(i), i);
            end
            accuracy = accuracy / size(label, 2);
        end
    end
    
    methods
        function obj = NeuralNetwork(layers)
            %NEURALNETWORK Construct a neural network.
            %   Construct a neural network with exact `layers`.
            obj.layers = layers;
            obj.outputLayer = Output(obj.layers{end}.outputSize);
        end
        
        function prediction = predict(obj, X)
            %PREDICT Predict the classes that the examples belong to.
            %   Predict the classes that the training examples in `X`
            %   belong to.
            input = X;
            for i = 1 : length(obj.layers)
                input = obj.layers{i}.forwardProp(input);
            end
            prediction = obj.outputLayer.forwardProp(input);
            %[conf, prediction] = max(prediction);
        end
        
        function train(obj, X, Y, batchSize, epochs)
            %TRAIN Train the neural network with provided data.
            %   Train the neural network with given training examples
            %   stored in (`X`, `Y`). `X` should be (_ * m) and `Y` should
            %   be (_ * m). `batchSize` determines how many training
            %   examples should be trained together.
            assert(ndims(X) == 4);
            assert(ismatrix(Y));
            assert(size(X, 4) == size(Y, 2));
            m = size(X, 4);
            assert(mod(m, batchSize) == 0);
            batchNum = m / batchSize;
            for e = 1 : epochs
                fprintf('epoches %d\n', e);
                batches = reshape(randperm(m), batchNum, batchSize);
                for b = 1 : batchNum
                    x = X(:, :, :, batches(b, :));
                    y = Y(:, batches(b, :));
                    obj.predict(x);
                    delta = obj.outputLayer.backProp(y);
                    for i = length(obj.layers) : -1 : 1
                        delta = obj.layers{i}.backProp(delta);
                    end
                end
            end
        end
    end
end

