import torch

def set_dtype(bits):
    if bits > 16:
        return torch.int32
    if bits > 8:
        return torch.int16
    else:
        return torch.int8
    
def ternary_quantize(tensor, bits, causal_mask=False):
    if causal_mask:
        lower_triangular = torch.tril(tensor)
        scale = lower_triangular.abs().mean().clamp(min=1e-5)
    else:
        scale = tensor.abs().mean().clamp(min=1e-5)
    result = (tensor / scale).round().clamp(-1, 1).to(dtype=torch.int8)
    return torch.tensor([0], device=tensor.device), scale, result
    
def calculate_quant_level(training, quant_scheduler, start_quant_level, full_quant_iter, iter_num):
    if not training:
        return 1
    if full_quant_iter == None:
        raise ValueError("Full quant iteration was not specified.")
    if iter_num == None:
        raise ValueError("Iter_num was not passed to GPT model")
    if quant_scheduler == "static":
        return start_quant_level
    elif quant_scheduler == "linear":
        return min(iter_num / full_quant_iter + (full_quant_iter * start_quant_level), 1)
    
def symmetric_quantize(tensor, bits, causal_mask=False):
    """
    Symmetric quantization function
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :return: zero point, scale, quantized tensor
    """
    bit_max = (1 << (bits - 1)) - 1
    bit_min = -bit_max - 1
    if causal_mask:
        # Apply torch.tril to get the lower triangular part (including diagonal)
        lower_triangular = torch.tril(tensor)

        # Find the maximum value
        abs_max = lower_triangular.abs().max()
    else:
        abs_max = tensor.abs().max()
    scale = abs_max / bit_max
    xi_array = torch.round(tensor / scale)
    clamped_array = torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))
    return torch.tensor([0], device=tensor.device), scale, clamped_array

def affine_quantize(tensor, bits):
    """
    Affine (asymmetric) quantization function
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :return: zero point, scale, quantized tensor
    """
    bit_max = (1 << (bits - 1)) - 1
    bit_min = -bit_max - 1
    max = tensor.max()
    min = tensor.min()
    scale = (max - min) / ((1 << bits) - 1)
    zero_point = -torch.round(min / scale) + bit_min
    xi_array = torch.round(tensor / scale) + zero_point
    return zero_point, scale, torch.clamp(xi_array, min=bit_min, max=bit_max).to(dtype=set_dtype(bits))

def stochastic_quantize(tensor, bits):
    """
    Stochastic quantization function
    :param tensor: Tensor to be quantized
    :param bits: Number of bits of quantization
    :return: zero point, scale, quantized tensor
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    # Steps:
    # Normalizes the tensor values to the range [0,𝑠]
    # Uses stochastic rounding to determine the quantized values.
    # Combines the quantized values with the original signs.
    # Returns the scaling factor and the quantized tensor.

    # maximum integer value that can be represented with the given number of bits. For example, if bits=8, s=255 (2^8-1)
    s = (1 << bits) - 1

    # norm = torch.norm(tensor)
    norm = tensor.abs().max()

    # captures the sign of each element in the tensor
    sign_array = torch.sign(tensor).to(dtype=torch.int8)

    # scales the absolute values of the tensor to the range [0,𝑠]
    l_array = torch.abs(tensor) / norm * s
    l_array_floored = l_array.to(dtype=torch.int)

    prob_array = l_array - l_array_floored
    # fractional part of l_array, clamped between 0 and 1 (rescaled so min is 0 and max is 1)
    prob_array = torch.clamp(prob_array, min=0.0, max=1.0)


    # stochastic rounding: draw 0 or 1s from a Bernoulli distribution with probability equal to the corresponding element
    mask = torch.bernoulli(prob_array)

    # final quantized array. Elements are incremented by 1 if the corresponding element in mask is 1 (stochastic rounding)
    xi_array = l_array_floored + mask
    xi_array = xi_array.to(dtype=torch.int32)

    # combines the sign and the quantized magnitude to get the final quantized tensor with the same sign as the original tensor
    sign_xi_array = (sign_array * xi_array).to(dtype=set_dtype(bits))
    norm = norm / s

    return torch.tensor([0], device=tensor.device), norm, sign_xi_array

def dequantize(zero_point, scale, tensor, causal_mask=False):
    """
    Dequantize the quantizated tensor
    :param zero_point: zero point of tensor
    :param scale: scale of tensor
    :param tensor: quantized tensor
    :return: Dequantized weights
    """
    dequantized = (tensor - zero_point) * scale
    return dequantized

def fake_quantize_act(obj, activation, tensor, num_bits, quant_method, iter_num, causal_mask=False):
    zero_point, scale, act = quantize_dictionary[quant_method](tensor, num_bits, causal_mask=causal_mask)
    setattr(obj, activation, act)
    setattr(obj, f"{activation}_scale", scale)
    setattr(obj, f"{activation}_zero_point", zero_point)
    dequantized = dequantize(zero_point, scale, act, causal_mask=causal_mask)
    if causal_mask:
        # Create a mask for the upper triangular part
        upper_tri_mask = torch.triu(torch.ones_like(tensor), diagonal=1).bool()

        # Set the upper triangular part to -inf
        tensor[upper_tri_mask] = 0

    # If scheduler is set, then we need to calculate the current quantization level
    if obj.quant_scheduler != None:
        quant_level = calculate_quant_level(obj.training, obj.quant_scheduler, obj.start_quant_level, obj.full_quant_iteration, iter_num)
        # print quantization level for every evaluation interval
        if obj.training and iter_num % obj.eval_interval == 0:
            print("quant level: ", quant_level)
        # adds quantization error to the original tensor
        result = tensor + quant_level * (dequantized - tensor).detach()
    else:
        result = dequantized

    if causal_mask:
        result[upper_tri_mask] = -float('inf')

    return result

class FakeLinearQuantizationFunction(torch.autograd.Function):
    """Simulates error caused by quantization. Uses Straight-Through Estimator for Back prop
    Source: https://github.com/Alexstrasza98/Transformer-Quantization/blob/main
    Source License: MIT
    """

    @staticmethod
    def forward(ctx, input, training, quant_scheduler, start_quant_level, full_quant_iter, eval_interval, steps, bits=7, quantization_method="affine_quant"):
        """
        Forward pass
        :param ctx: Context object to store information for the backward pass (not used in this case)
        :param input: The input tensor to be quantized
        :param bits: The number of bits for quantization (default is 7)
        :return: Dequantized tensor
        """
        # steps:
        # Quantize the input tensor using the quantize function.
        # Dequantize the quantized values using the dequantize function.
        # Return the dequantized tensor, which approximates the input tensor but includes the quantization error.
        zero_point, norm, quantized_weight = quantize_dictionary[quantization_method](input, bits)
        # If scheduler is set, then we need to calculate the current quantization level
        dequantized = dequantize(zero_point, norm, quantized_weight)
        if quant_scheduler != None:
            quant_level = calculate_quant_level(training, quant_scheduler, start_quant_level, full_quant_iter, steps)
            if training and steps % eval_interval == 0:
                print("quant level: ", quant_level)
            
            return input + quant_level * (dequantized - input).detach()
        return dequantized

    @staticmethod
    def backward(ctx, grad_output):
        # Straight-Through Estimator (STE): passes grad_output through as the gradient with respect to the input
        # gradient is approximated by simply passing the gradient from the output directly to the input, 
        # ignoring the quantization operation
        return grad_output, None, None, None, None, None, None, None, None

quantize_dictionary = {
    "ternary_quant": ternary_quantize,
    "symmetric_quant": symmetric_quantize,
    "affine_quant": affine_quantize,
    "stochastic_quant": stochastic_quantize
}

_fake_quantize = FakeLinearQuantizationFunction.apply
