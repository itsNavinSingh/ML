import torch
import conv_block
import inception_block

class InceptionNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # (n, 3, 224, 224)
        self.conv1 = conv_block.ConvBlock(
            in_channel = 3,
            out_channel = 64,
            kernel_size = (7, 7),
            stride = 2,
            padding = 3
        )
        # (n, 64, 112, 112)
        self.pool1 = torch.nn.MaxPool2d(
            kernel_size = (3, 3),
            stride = 2,
            padding=1
        )
        # (n, 64, 56, 56)
        self.conv2 = conv_block.ConvBlock(
            in_channel=64,
            out_channel=192,
            kernel_size=(3, 3),
            stride=1,
            padding=1
        )
        # (n, 192, 56, 56)
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2,
            padding=1
        )
        # (n, 192, 28, 28)
        self.inception3a = inception_block.InceptionBlock(
            in_feature=192,
            out_1_1=64,
            pre_3_3=96,
            out_3_3=128,
            pre_5_5=16,
            out_5_5=32,
            out_maxpool=32
        )
        # (n, 64 + 128 + 32 + 32 = 256, 28, 28)
        self.inception3b = inception_block.InceptionBlock(
            in_feature=256,
            out_1_1=128,
            pre_3_3=128,
            out_3_3=192,
            pre_5_5=32,
            out_5_5=96,
            out_maxpool=64
        )
        # (n, 128 + 192 + 96, 64 = 480, 28, 28)
        self.pool3 = torch.nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2,
            padding=1
        )
        # (n, 480, 14, 14)
        self.inception4a = inception_block.InceptionBlock(
            in_feature=480,
            out_1_1=192,
            pre_3_3=96,
            out_3_3=208,
            pre_5_5=16,
            out_5_5=48,
            out_maxpool=64
        )
        # (n, 192 + 208 + 48 + 64 = 512, 14, 14)
        self.inception4b = inception_block.InceptionBlock(
            in_feature=512,
            out_1_1=160,
            pre_3_3=112,
            out_3_3=224,
            pre_5_5=24,
            out_5_5=64,
            out_maxpool=64
        )
        # (n, 160 + 224 + 64 + 64 = 512, 14, 14)
        self.inception4c = inception_block.InceptionBlock(
            in_feature=512,
            out_1_1=128,
            pre_3_3=128,
            out_3_3=256,
            pre_5_5=24,
            out_5_5=64,
            out_maxpool=64
        )
        # (n, 128 + 256 + 64 + 64 = 512, 14, 14)
        self.inception4d = inception_block.InceptionBlock(
            in_feature=512,
            out_1_1=112,
            pre_3_3=144,
            out_3_3=288,
            pre_5_5=32,
            out_5_5=64,
            out_maxpool=64
        )
        # (n, 112 + 288 + 64 + 64 = 528, 14, 14)
        self.inception4e = inception_block.InceptionBlock(
            in_feature=528,
            out_1_1=256,
            pre_3_3=160,
            out_3_3=320,
            pre_5_5=32,
            out_5_5=128,
            out_maxpool=128
        )
        # (n, 256 + 320 + 128 + 128 = 832, 14, 14)
        self.pool4 = torch.nn.MaxPool2d(
            kernel_size=(3, 3),
            stride=2,
            padding=1
        )
        # (n, 832, 7, 7)
        self.inception5a = inception_block.InceptionBlock(
            in_feature=832,
            out_1_1=256,
            pre_3_3=160,
            out_3_3=320,
            pre_5_5=32,
            out_5_5=128,
            out_maxpool=128
        )
        # (n, 256 + 320 + 128 + 128 = 832, 7, 7)
        self.inception5b = inception_block.InceptionBlock(
            in_feature=832,
            out_1_1=384,
            pre_3_3=192,
            out_3_3=384,
            pre_5_5=48,
            out_5_5=128,
            out_maxpool=128
        )
        # (n, 384 + 384 + 128 + 128 = 1024, 7, 7)
        self.pool5 = torch.nn.AvgPool2d(
            kernel_size=(7, 7),
            stride=1,
            padding=0
        )
        # (n, 1024, 1, 1) -> flatten -> (n, 1024)
        self.dropout = torch.nn.Dropout(0.4)
        self.fc = torch.nn.Linear(
            in_features=1024,
            out_features=1000
        )
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x