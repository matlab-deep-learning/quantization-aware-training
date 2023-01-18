classdef QuantizedConvolutionBatchNormTrainingLayer < nnet.layer.Layer & nnet.layer.Formattable
    %% QuantizedConvolutionBatchNormTrainingLayer Layer for Quantization Aware Training
    %   This custom layer introduces quantization error to a
    %   fused convolution layer and a batch normalization layer during
    %   training.

    % Copyright 2023 The Mathworks, Inc.


    properties (Learnable)
        Network
    end

    methods
        function obj = QuantizedConvolutionBatchNormTrainingLayer(cLayer, bLayer)
            % Freeze the Scale and Offset Learn Factor of the
            % BatchNormalizaiton Layer so to use the statistics collected
            % at training of the original network
            bLayer.ScaleLearnRateFactor = 0;
            bLayer.OffsetLearnRateFactor = 0;

            % Construct a dlnetwork as the Learnable of this custom layer
            obj.Network = dlnetwork([cLayer bLayer], 'Initialize', false);

            obj.Name = cLayer.Name;
            obj.Description = "Quantization Aware Conv-BN Layer Group for Training";
            obj.Type = "Quantized Fused Convolution Layer";
        end

        function Z = predict(layer, X)
            % Call predict on the underlying network if the network is not
            % yet initialized to avoid errors in inspecting the LayerGraph
            % before training.
            if ~layer.Network.Initialized
                Z = predict(layer.Network, X);
                return;
            end

            % Calculate the adjusted Weights and Bias of the convolution
            % layer in the underlying network during fusion.
            [adjustedWeights, adjustedBias] = foldBatchNormalizationParameters(layer.Network);

            % Quantize adjusted Weights to float.
            adjustedWeights = quantizeToFloat(adjustedWeights);

            % Recrete the learnables table using the adjusted Weights and
            % Bias.
            newLearnables = layer.Network.Learnables;
            newLearnables.Value{1} = adjustedWeights;
            newLearnables.Value{2} = adjustedBias;

            % Set learnables back on the Network.
            layer.Network.Learnables = newLearnables;

            % Call predict on the underlying Network tapping the
            % activations of the convolution layer only since the
            % batchNormalization has already been applied during the fusion
            % of foldBatchNormalizationParameters.
            Z = predict(layer.Network, X, 'Outputs', layer.Name);

            % Quantize the activation to flaot.
            Z = quantizeToFloat(Z);
        end

    end

end
