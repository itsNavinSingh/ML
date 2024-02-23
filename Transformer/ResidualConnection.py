import torch
class ResidualConnection(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dropout = torch.nn.Dropout(config['dropout'])
    self.eps = config['eps']
    self.alpha = torch.nn.Parameter(torch.ones(1))
    self.bias = torch.nn.Parameter(torch.zeros(1))
  def forward(self, previousOutput, skipOutput):
    x = self.dropout(previousOutput + skipOutput)
    mean = x.mean(dim = -1, keepdim=True)
    std = x.std(dim = -1, keepdim=True)
    return self.alpha * (x-mean) / (std+self.eps) + self.bias