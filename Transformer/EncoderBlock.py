import torch
from MultiHeadAttention import MultiheadAttentionLayer
from ResidualConnection import ResidualConnection
from FeedForward import FFN
class EncoderBlock(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.multihead = MultiheadAttentionLayer(config)
    self.addNorm1 = ResidualConnection(config)
    self.ffn = FFN(config)
    self.addNorm2 = ResidualConnection(config)
  def forward(self, x):
    out = self.multihead(x, x, x)
    x = self.addNorm1(out, x)
    out = self.ffn(x)
    return self.addNorm2(out, x)