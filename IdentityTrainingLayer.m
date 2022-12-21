classdef IdentityTrainingLayer < nnet.layer.Layer
    %% IdentityTrainingLayer that returns the input as output

    % Copyright 2022 The Mathworks, Inc.
    
    methods

        function obj = IdentityTrainingLayer(originalLayer)
            obj.Name = originalLayer.Name;
            obj.Type = "Identity Training Layer";
            obj.Description = "No operation to forward behavior";
        end


        function X = predict(layer, X)
            % No op - return the input as output 
        end

    end

end

