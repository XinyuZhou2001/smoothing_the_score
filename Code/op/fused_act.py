import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        if bias:
            self.bias = nn.Parameter(torch.zeros(channel))
        else:
            self.bias = None
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        bias_shape = [1, bias.shape[0]] + rest_dim
        bias_expanded = bias.view(*bias_shape)
        output = F.leaky_relu(input + bias_expanded, negative_slope=negative_slope)
    else:
        output = F.leaky_relu(input, negative_slope=negative_slope)
    return output * scale