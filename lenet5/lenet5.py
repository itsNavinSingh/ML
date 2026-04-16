import torch

class LeNet5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # (1, 32, 32)
        self.conv1 = torch.nn.Conv2d(
            in_channels = 1,
            out_channels = 6,
            kernel_size = (5, 5),
            stride = 1,
            padding = 0,
        )
        # (6, 28, 28)
        self.pool1 = torch.nn.AvgPool2d(
            kernel_size = 2,
            stride=2,
            padding=0,
        )
        self.activ1 = torch.nn.Tanh()
        # (6, 14, 14)

        self.conv2 = torch.nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5, 5),
            stride=1,
            padding=0,
        )
        # (16, 10, 10)
        self.pool2 = torch.nn.AvgPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.activ2 = torch.nn.Tanh()
        # (16, 5, 5)

        self.conv3 = torch.nn.Conv2d(
            in_channels=16,
            out_channels=120,
            kernel_size=(5, 5),
            stride=1,
            padding=0,
        )
        self.activ3 = torch.nn.Tanh()
        # (120, 1, 1) -> flatten -> 120

        self.fnn1 = torch.nn.Linear(
            in_features=120,
            out_features=84,
        )
        self.activ4 = torch.nn.Tanh()

        #(84)
        self.fnn2 = torch.nn.Linear(
            in_features=84,
            out_features=10,
        )
    def forward(self, x):
        x = self.activ1(self.pool1(self.conv1(x)))
        x = self.activ2(self.pool2(self.conv2(x)))
        x = self.activ3(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.activ4(self.fnn1(x))
        x = self.fnn2(x)
        return x
