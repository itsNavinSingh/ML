import torch
import math
class MaskedMultiheadAttentionLayer(torch.nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dModel = config['dModel']
    self.h = config['h']
    assert self.dModel%self.h == 0, "dModel is not divisible by h"
    self.dK = self.dModel//self.h
    self.wK = torch.nn.Linear(self.dModel, self.dModel)
    self.wQ = torch.nn.Linear(self.dModel, self.dModel)
    self.wV = torch.nn.Linear(self.dModel, self.dModel)
    mask = torch.triu(torch.ones(config['seqlen'], config['seqlen']))
    self.mask = mask.masked_fill(mask == 1, float('-inf'))
    self.wO = torch.nn.Linear(self.dModel, self.dModel)
    self.dropout = torch.nn.Dropout(config['dropout'])
  def forward(self, query, key, value):
    query = self.wQ(query)
    key = self.wK(key)
    value = self.wV(value)
    query = query.view(query.shape[0], query.shape[1], self.h, self.dK).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.dK).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.dK).transpose(1, 2)
    attentionScores = (query @ key.transpose(-2, -1)) / math.sqrt(self.dK)
    attentionScores = attentionScores + self.mask
    attentionScores = attentionScores.softmax(dim = -1)
    attentionScores = self.dropout(attentionScores)
    x = attentionScores @ value
    x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h*self.dK)
    return self.wO(x)