import torch
from EncoderBlock import EncoderBlock
class Encoder(torch.nn.Module):
  def __init__(self, dModel: int, h: int, dropout: float, internalFfnLen: int, encoderNumber: int, eps: float = 1e-6):
    super().__init__()
    self.encoderNumber = encoderNumber
    self.encoders = torch.nn.ModuleList([EncoderBlock(dModel, h, dropout, eps, internalFfnLen) for _ in range(encoderNumber)])
  def forward(self, x):
    for i in range(self.encoderNumber):
      x = self.encoders[i](x)
    return x