classdef QuantizedConvolutionTrainingLayer < nnet.layer.Layer & nnet.layer.Formattable
    %% QuantizedConvolutionTrainingLayer Layer for Quantization Aware Training
    %   This custom layer introduces quantization error to a
    %   convolution layer during training.

    % Copyright 2023 The Mathworks, Inc.


    properties (Learnable)
        Network
    end

    methods
        function obj = QuantizedConvolutionTrainingLayer(cLayer)
            % Construct a dlnetwork as the Learnable of this custom layer
            obj.Network = dlnetwork(cLayer, 'Initialize', false);

            obj.Name = cLayer.Name;
            obj.Description = "Quantization Aware Conv Layer for Training";
            obj.Type = "Quantized Convolution Layer";
        end

        function Z = predict(layer, X)
            % Call predict on the underlying network if the network is not
            % yet initialized to avoid errors in inspecting the LayerGraph
            % before training.
            if ~layer.Network.Initialized
                Z = predict(layer.Network, X);
                return;
            end

            % Capture the Weights of the convolution
            % layer in the underlying network
            weights = layer.Network.Learnables.Value{1};

            % Quantize the Weights to float.
            weights = quantizeToFloat(weights);

            % Set learnables back on the Network.
            layer.Network.Learnables.Value{1} = weights;

            % Call predict on the underlying Network
            Z = predict(layer.Network, X);

            % Quantize the activation to float.
            Z = quantizeToFloat(Z);
        end

    end

end