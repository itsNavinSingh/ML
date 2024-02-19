import torch
class FFN(torch.nn.Module):
  def __init__(self, dModel: int, internalLen: int, dropout: float):
    super().__init__()
    self.linear1 = torch.nn.Linear(dModel, internalLen)
    self.dropout = torch.nn.Dropout(dropout)
    self.linear2 = torch.nn.Linear(internalLen, dModel)
  def forward(self, x):
    x = torch.relu(self.linear1(x))
    x = self.linear2(self.dropout(x))
    return x