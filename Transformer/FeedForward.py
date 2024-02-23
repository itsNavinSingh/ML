import torch
class FFN(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.linear1 = torch.nn.Linear(config['dModel'], config['internalFfnLen'])
    self.dropout = torch.nn.Dropout(config['dropout'])
    self.linear2 = torch.nn.Linear(config['internalFfnLen'], config['dModel'])
  def forward(self, x):
    x = torch.relu(self.linear1(x))
    x = self.linear2(self.dropout(x))
    return x