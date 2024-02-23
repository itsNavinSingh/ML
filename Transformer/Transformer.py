import torch
from Encoder import Encoder
from Decoder import Decoder
from Embedding import Embedding
from PositionalEncoding import PositionalEncoding
class Transformer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.srcEmbedding = Embedding(config['dModel'], config["srcVocabSize"])
        self.tgtEmbedding = Embedding(config['dModel'], config["tgtVocabSize"])
        self.srcPositionalEncoding = PositionalEncoding(config['dModel'], config['seqLen'], config['dropout'])
        self.tgtPositionalEncoding = PositionalEncoding(config['dModel'], config['seqLen'], config['dropout'])
        self.linear = torch.nn.Linear(config['dModel'], config["tgtVocabSize"])
    def forward(self, incoderInput, decoderInput):
        x = self.srcEmbedding(incoderInput)
        x = self.srcPositionalEncoding(x)
        x = self.encoder(x)
        y = self.tgtEmbedding(decoderInput)
        y = self.tgtPositionalEncoding(y)
        y = self.decoder(x, y)
        y = self.linear(y)
        return torch.log_softmax(y, dim=-1)