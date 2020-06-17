import torch.nn.functional as F
import torch.nn as nn

def crossEntropy_loss(output, target):
    return nn.CrossEntropyLoss()(output, target)

def nll_loss(output, target):
    return F.nll_loss(output, target)
