import torch
from EncoderBlock import EncoderBlock
class Encoder(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.encoderNumber = config['EncoderNo']
    self.encoders = torch.nn.ModuleList([EncoderBlock(config) for _ in range(self.encoderNumber)])
  def forward(self, x):
    for i in range(self.encoderNumber):
      x = self.encoders[i](x)
    return x