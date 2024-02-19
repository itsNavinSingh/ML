import torch
import math
class PositionalEncoding(torch.nn.Module):
  def __init__(self, dModel: int, seqLen: int, dropout: float):
    super().__init__()
    self.dropout = torch.nn.Dropout(dropout)
    self.positionalEncoding = torch.zeros(seqLen, dModel)
    position = torch.arange(0, seqLen, dtype=torch.float).unsqueeze(1)
    divTerm = torch.exp(torch.arange(0, dModel, 2).float() * (-math.log(10000.0)/dModel))
    self.positionalEncoding[:, 0::2] = torch.sin(position * divTerm)
    self.positionalEncoding[:, 1::2] = torch.cos(position * divTerm)
    self.positionalEncoding = self.positionalEncoding.unsqueeze(0)
    self.positionalEncoding.requires_grad = False
  def forward(self, x):
    x = x + (self.positionalEncoding[:, :x.shape[1], :])
    return self.dropout(x)