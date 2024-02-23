import torch
from MaskedMultiHeadAttention import MaskedMultiheadAttentionLayer
from ResidualConnection import ResidualConnection
from MultiHeadAttention import MultiheadAttentionLayer
from FeedForward import FFN
class DecoderBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.maskMultiHead = MaskedMultiheadAttentionLayer(config)
        self.addNorm1 = ResidualConnection(config)
        self.multiHead = MultiheadAttentionLayer(config)
        self.addNorm2 = ResidualConnection(config)
        self.ffn = FFN(config)
        self.addNorm3 = ResidualConnection(config)
    def forward(self, EncoderInput, DecoderInput):
        out = self.maskMultiHead(DecoderInput, DecoderInput, DecoderInput)
        x = self.addNorm1(out, DecoderInput)
        out = self.multiHead(EncoderInput, EncoderInput, x)
        x = self.addNorm2(out, x)
        out = self.ffn(x)
        return self.addNorm3(out, x)
