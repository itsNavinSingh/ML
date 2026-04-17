import torch

class AlexNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # (n, 3, 227, 227)
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=96,
            kernel_size=(11, 11),
            stride=4,
            padding=0,
        )
        # (n, 96, 55, 55)
        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )
        self.activ1 = torch.nn.ReLU()
        # (n, 96, 27, 27)
        self.conv2 = torch.nn.Conv2d(
            in_channels=96,
            out_channels=256,
            kernel_size=(5, 5),
            stride=1,
            padding=2,
        )
        # (n, 256, 27, 27)
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )
        self.activ2 = torch.nn.ReLU()
        # (n, 256, 13, 13)
        self.conv3 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        # (n, 384, 13, 13)
        self.activ3 = torch.nn.ReLU()
        # (n, 384, 13, 13)
        self.conv4 = torch.nn.Conv2d(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.activ4 = torch.nn.ReLU()
        # (n, 384, 13, 13)
        self.conv5 = torch.nn.Conv2d(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        # (n, 256, 13, 13)
        self.pool5 = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )
        self.activ5 = torch.nn.ReLU()
        # (n, 256, 6, 6) -> flatten -> (n, 9216)
        self.fnn6 = torch.nn.Linear(
            in_features=9216,
            out_features=4096,
        )
        self.dropout6 = torch.nn.Dropout(0.5)
        self.activ6 = torch.nn.ReLU()
        # (n, 4096)
        self.fnn7 = torch.nn.Linear(
            in_features=4096,
            out_features=4096,
        )
        self.dropout7 = torch.nn.Dropout(0.5)
        self.activ7 = torch.nn.ReLU()
        # (n, 4096)
        self.fnn8 = torch.nn.Linear(
            in_features=4096,
            out_features=1000,
        )
        # (n, 1000)
        return
    def forward(self, x):
        x = self.activ1(self.pool1(self.conv1(x)))
        x = self.activ2(self.pool2(self.conv2(x)))
        x = self.activ3(self.conv3(x))
        x = self.activ4(self.conv4(x))
        x = self.activ5(self.pool5(self.conv5(x)))
        x = x.view(x.size(0), -1)
        x = self.activ6(self.dropout6(self.fnn6(x)))
        x = self.activ7(self.dropout7(self.fnn7(x)))
        x = self.fnn8(x)
        return x
