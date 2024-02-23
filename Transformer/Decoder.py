import torch
from DecoderBlock import DecoderBlock
class Decoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.decoderNumber = config['DecoderNo']
        self.decoders = torch.nn.ModuleList([DecoderBlock(config) for _ in range(self.decoderNumber)])
    def forward(self, EncoderInput, DecoderInput):
        for i in range(self.decoderNumber):
            DecoderInput = self.decoders[i](EncoderInput, DecoderInput)
        return DecoderInput