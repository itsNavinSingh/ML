import torch
import math
class Embedding(torch.nn.Module):
  def __init__(self, dModel: int, vocabSize: int):
    super().__init__()
    self.dModel = dModel
    self.vocabSize = vocabSize
    self.embedding = torch.nn.Embedding(vocabSize, dModel)
  def forward(self, x):
    return self.embedding(x) * math.sqrt(self.dModel)