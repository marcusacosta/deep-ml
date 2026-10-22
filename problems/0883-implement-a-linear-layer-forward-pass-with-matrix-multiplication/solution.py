import torch

def linear_forward(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # TODO: implement y = x W^T + b using PyTorch ops
    transposed = torch.transpose(W, 0, 1)
    res = x @ transposed + b
    
    return res
