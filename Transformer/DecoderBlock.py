import torch
from MaskedMultiHeadAttention import MaskedMultiheadAttentionLayer
from ResidualConnection import ResidualConnection
from MultiHeadAttention import MultiheadAttentionLayer
from FeedForward import FFN
class DecoderBlock(torch.nn.Module):
    def __init__(self, dModel: int, seqLen: int, h: int, dropout: float, eps: float, internalFfnLen: int):
        super().__init__()
        self.maskMultiHead = MaskedMultiheadAttentionLayer(dModel, seqLen, h, dropout)
        self.addNorm1 = ResidualConnection(dropout, eps)
        self.multiHead = MultiheadAttentionLayer(dModel, h, dropout)
        self.addNorm2 = ResidualConnection(dropout, eps)
        self.ffn = FFN(dModel, internalFfnLen, dropout)
        self.addNorm3 = ResidualConnection(dropout, eps)
    def forward(self, EncoderInput, DecoderInput):
        out = self.maskMultiHead(DecoderInput, DecoderInput, DecoderInput)
        x = self.addNorm1(out, DecoderInput)
        out = self.multiHead(EncoderInput, EncoderInput, x)
        x = self.addNorm2(out, x)
        out = self.ffn(x)
        return self.addNorm3(out, x)
