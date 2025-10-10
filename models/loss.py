import torch

def Loss(*args):
    mae_loss = torch.nn.L1Loss()
    output = mae_loss(args[0], args[1])
    return output