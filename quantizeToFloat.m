function value = quantizeToFloat(value)
    %% quantizeToFloat Quantizes a value and rescales back to floating point
    %
    % quantizedValue = quantizeToFloat(value) returns a floating point
    % value that has been quantized using best precision scaling.
    %
    %   Example:
    %        quantizedValue = quantizeToFloat(dlarray(single(365.247)))

    % Copyright 2023 The Mathworks, Inc.

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