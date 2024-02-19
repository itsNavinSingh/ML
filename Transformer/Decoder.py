import torch
from DecoderBlock import DecoderBlock
class Decoder(torch.nn.Module):
    def __init__(self, dModel: int, seqLen: int, h: int, dropout: float, eps: float, internalFfnLen: int, decoderNmber: int):
        super().__init__()
        self.decoderNumber = decoderNmber
        self.decoders = torch.nn.ModuleList([DecoderBlock(dModel, seqLen, h, dropout, eps, internalFfnLen) for _ in range(decoderNmber)])
    def forward(self, EncoderInput, DecoderInput):
        for i in range(self.decoderNumber):
            DecoderInput = self.decoders[i](EncoderInput, DecoderInput)
        return DecoderInput