import torch
from MultiHeadAttention import MultiheadAttentionLayer
from ResidualConnection import ResidualConnection
from FeedForward import FFN
class EncoderBlock(torch.nn.Module):
  def __init__(self, dModel: int, h: int, dropout: float, eps: float, internalFfnLen: int):
    super().__init__()
    self.multihead = MultiheadAttentionLayer(dModel, h, dropout)
    self.addNorm1 = ResidualConnection(dropout, eps)
    self.ffn = FFN(dModel, internalFfnLen, dropout)
    self.addNorm2 = ResidualConnection(dropout, eps)
  def forward(self, x):
    out = self.multihead(x, x, x)
    x = self.addNorm1(out, x)
    out = self.ffn(x)
    return self.addNorm2(out, x)