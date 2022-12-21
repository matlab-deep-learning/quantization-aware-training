%  foldBatchNormalizationParameters Adjusts Convolution Learnables for Fusion
%  Calculates the adjusted learnables of a convolution layer from a
%  would-be fusion with a batch normalization layer.
%  
%  [ADJUSTEDWEIGHTS, ADJUSTEDBIAS] = foldBatchNormalizationParameters(NET) a dlnetwork with a Convolution2D or GroupedConvolution2D layer
%  as the first layer and BatchNormalization layer as the second layer,
%  return the adjusted weights and bias of the convolution layer
%