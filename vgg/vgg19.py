import torch

class VGG19(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv2 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv3 = torch.nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv4 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv5 = torch.nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv6 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv7 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv8 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv9 = torch.nn.Conv2d(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv10 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv11 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv12 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv13 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv14 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv15 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )
        self.conv16 = torch.nn.Conv2d(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
        )

        self.pool1 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.pool2 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.pool3 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.pool4 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        self.pool5 = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = torch.nn.Linear(
            in_features=25088,
            out_features=4096,
        )
        self.fc2 = torch.nn.Linear(
            in_features=4096,
            out_features=4096,
        )
        self.fc3 = torch.nn.Linear(
            in_features=4096,
            out_features=1000,
        )

        self.activ1 = torch.nn.ReLU()
        self.activ2 = torch.nn.ReLU()
        self.activ3 = torch.nn.ReLU()
        self.activ4 = torch.nn.ReLU()
        self.activ5 = torch.nn.ReLU()
        self.activ6 = torch.nn.ReLU()
        self.activ7 = torch.nn.ReLU()
        self.activ8 = torch.nn.ReLU()
        self.activ9 = torch.nn.ReLU()
        self.activ10 = torch.nn.ReLU()
        self.activ11 = torch.nn.ReLU()
        self.activ12 = torch.nn.ReLU()
        self.activ13 = torch.nn.ReLU()
        self.activ14 = torch.nn.ReLU()
        self.activ15 = torch.nn.ReLU()
        self.activ16 = torch.nn.ReLU()
        self.activ17 = torch.nn.ReLU()
        self.activ18 = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.activ1(self.conv1(x))
        x = self.activ2(self.conv2(x))
        x = self.pool1(x)
        x = self.activ3(self.conv3(x))
        x = self.activ4(self.conv4(x))
        x = self.pool2(x)
        x = self.activ5(self.conv5(x))
        x = self.activ6(self.conv6(x))
        x = self.activ7(self.conv7(x))
        x = self.activ8(self.conv8(x))
        x = self.pool3(x)
        x = self.activ9(self.conv9(x))
        x = self.activ10(self.conv10(x))
        x = self.activ11(self.conv11(x))
        x = self.activ12(self.conv12(x))
        x = self.pool4(x)
        x = self.activ13(self.conv13(x))
        x = self.activ14(self.conv14(x))
        x = self.activ15(self.conv15(x))
        x = self.activ16(self.conv16(x))
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.activ17(self.fc1(x))
        x = self.activ18(self.fc2(x))
        x = self.fc3(x)
        return x