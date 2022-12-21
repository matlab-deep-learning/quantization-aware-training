classdef QuantizedConvolutionBatchNormTrainingLayer < nnet.layer.Layer & nnet.layer.Formattable
    %% QUANTIZEDCONVOLUTIONBATCHNORMTRAININGLAYER Layer for Quantization Aware Training
    %   This custom layer takes a emulates the quantization effects of a
    %   fused convolution layer and a batch normalization layer during
    %   training.

    % Copyright 2022 The Mathworks, Inc.


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

            % Quantize and unquantize the adjusted Weights.
            adjustedWeights = boomerangQuantize(adjustedWeights);

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

            % Quantize and unquantize the activation of the Layer.
            Z = boomerangQuantize(Z);
        end

    end

end

function value = boomerangQuantize(value)
    % Do quantize followed by unquantize operations.

    % Calculate the ideal scaling factor using the input range.
    m = extractdata(gather(max(abs(value(:)))));
    scalingFactor = double(floor(log2(m)));
    % Adjust the scaling factor by 6. 8 bit wordlength - 1
    % sign - 1 floor
    scalingFactor = scalingFactor - 6;

    % Scale the value using the calculated scaling factor.
    value = scaleValue(value, scalingFactor);

    % Saturate to int8 range.
    value = saturateValue(value);

    % Round values while bypassing the dlgradient calculation.
    value = bypassdlgradients(@round, value);

    % Rescale values to single range.
    value = rescaleValue(value, scalingFactor);
end

function value = scaleValue(value, scalingFactor)
    % Scale the value using the calculated scaling factor.
    value = value*single((2^( -1*scalingFactor )));
end

function value = saturateValue(value)
    % Saturate to int8 range
    value = max(value,-128); % intmin('int8')
    value = min(value, 127); % intmax('int8')
end

function value = rescaleValue(value, scalingFactor)
    % Rescale values to single range.
    value = value*single(2^scalingFactor);
end
