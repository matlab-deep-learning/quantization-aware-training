# Quantization Aware Training with MobileNet-v2

This example shows how to perform quantization aware training as a way to prepare a network for quantization. Quantization aware training is a method that can help recover accuracy lost due to quantizing a network to use 8-bit scaled integer weights and biases. Networks like MobileNet-v2 are especially sensitive to quantization due to the significant variation in range of values of the weight tensor of the convolution and grouped convolution layers.

This example shows how pre-processing a network with quantization aware training can produce a quantized network with accuracy on par with the original unquantized network.

| Network      | Accuracy |
| ----------- | ----------- |
| Original network      | **0.8965**       |
| `int8` network via post-training quantization   | 0.2943        |
| `int8` network via quantization aware training   | **0.8910**        |

## **Running the Example**

Open and run the live script `QuantizationAwareTrainingWithMobilenetv2.mlx`.

Additional files:

- `QuantizedConvolutionBatchNormTrainingLayer`: implements boomerang quantization using best precision scaling for the convolution layer's weights and activations.
- `IdentitytTrainingLayer`: no-op layer that acts as a placeholder for batch normalization layers.
- `bypassdlgradients`: function to perform straight through estimation for a given operation.
- `foldBatchNormalizationParameters`: function to calculate the adjusted weights and bias for dlnetwork that contains a convolution layer followed by a batch normalization layer.

### Requirements:

- [MATLAB](https://www.mathworks.com/products/matlab.html) version R2022b or later
- [Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html)
- [Deep Learning Toolbox Model Quantization Library](https://www.mathworks.com/matlabcentral/fileexchange/74614-deep-learning-toolbox-model-quantization-library)

## About Quantization Aware Training

This example shows the steps of a quantization workflow:

- _Preparing_ a floating point network trained on the flowers data set.
- _Training_ the network with custom layers that introduce quantization error during training.
- _Evaluating_ the performance of the fully quantized model and comparing it to the performance of original method.

To begin, create a transfer learned MobileNet-v2 using the flower dataset.

During training, replace the convolution layers with custom training layers that perform pseudo-quantization operations on the weights and activations of the layer. For each forward training pass, values at quantization aware convolution layers are quantized and then unquantized. This quantizing and unquantizing can be referred to as "boomerang quantization". An example of a boomerang quantization operation would take a value like `365.247` and quantize to an integer representation of `91`. The integer value of `91` is then rescaled back to `364` introducing a quantization error of `-1.247`.
![Boomerang Quantize](./images/boomerangQuantize.png)

The quantization step uses a non-differentiable operation `round` that would normally break the training workflow by zeroing out the gradients. During quantization aware training, bypass the gradient calculations for non-differentiable operations using an identity function. The diagram below \[2\] shows how the identity function calculates the gradients for non-differentiable operations.

![Straight Through Estimation](./images/ste.png)

After training, the network returned from the `trainNetwork` function still has the quantization aware training layers. Replace the quantization aware training operators with operators that are specific to inference. Whereas the training graph operates on pseudo-quantized 32-bit floating-point values, in the inference graph, the network applies the convolution using `int8` inputs and weights.

| Conovolution Operation Graph at Training   | Convolution Operation Graph at Inference |
| ----------- | ----------- |
| ![Quantized operators during training](./images/quantized_training.png)   | ![Quantized operators during inference](./images/quantized_inference.png)|

## **References**

1. The TensorFlow Team. Flowers [http://download.tensorflow.org/example_images/flower_photos.tgz](http://download.tensorflow.org/example_images/flower_photos.tgz)
2. Gholami, A., Kim, S., Dong, Z., Mahoney, M., & Keutzer, K. (2021). A Survey of Quantization Methods for Efficient Neural Network Inference. Retrieved from [https://arxiv.org/abs/2103.13630](https://arxiv.org/abs/2103.13630)
3. Jacob, B., Kligys, S., Chen, B., Zhu, M., Tang, M., Howard, A., Adam, H., & Kalenichenko, D. (2017). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. Retrieved from [https://arxiv.org/abs/1712.05877](https://arxiv.org/abs/1712.05877)

Copyright 2022 The MathWorks, Inc.

[![View Quantization-Aware-Training-with-Mobilenetv2 on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://www.mathworks.com/matlabcentral/fileexchange/74614-deep-learning-toolbox-model-quantization-library)<!---TODO: update link-->
