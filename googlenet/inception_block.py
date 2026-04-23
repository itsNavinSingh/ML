import torch
import conv_block

class InceptionBlock(torch.nn.Module):
    def __init__(self, in_feature, out_1_1, pre_3_3, out_3_3, pre_5_5, out_5_5, out_maxpool):
        super().__init__()
        self.branch1 = conv_block.ConvBlock(
            in_channel=in_feature,
            out_channel=out_1_1,
            kernel_size = (1, 1),
            stride=1,
            padding=0
        )
        self.branch2 = torch.nn.Sequential(
            conv_block.ConvBlock(
                in_channel=in_feature,
                out_channel=pre_3_3,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            ),
            conv_block.ConvBlock(
                in_channel=pre_3_3,
                out_channel=out_3_3,
                kernel_size = (3, 3),
                stride=1,
                padding=1
            )
        )

        self.branch3 = torch.nn.Sequential(
            conv_block.ConvBlock(
                in_channel=in_feature,
                out_channel=pre_5_5,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            ),
            conv_block.ConvBlock(
                in_channel=pre_5_5,
                out_channel=out_5_5,
                kernel_size=(5, 5),
                stride=1,
                padding=2
            )
        )

        self.branch4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(
                kernel_size=(3, 3),
                stride=1,
                padding=1
            ),
            conv_block.ConvBlock(
                in_channel=in_feature,
                out_channel=out_maxpool,
                kernel_size=(1, 1),
                stride=1,
                padding=0
            )
        )
    def forward(self, x):
        return torch.cat([
            self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        ], 1)
    
class AuxInceptionBlock(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super(AuxInceptionBlock, self).__init__()
        self.pool = torch.nn.MaxPool2d(
            kernel_size=(5, 5),
            stride=(3, 3)
        )
        self.conv = conv_block.ConvBlock(
            in_channel=in_feature,
            out_channel=128,
            kernel_size=(1, 1)
        )
        self.fc1 = torch.nn.Linear(
            in_features=2048,
            out_features=1024
        )
        self.activ = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.7)
        self.fc2 = torch.nn.Linear(
            in_features=1024,
            out_features=out_feature
        )
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x