import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            **kwargs
        )
        self.norm = torch.nn.BatchNorm2d(num_features=out_channel)
        self.activ = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activ(x)
        return x
