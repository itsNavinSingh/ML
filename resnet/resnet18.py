import torch

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=False,
            **kwargs
        )
        self.bnorm = torch.nn.BatchNorm2d(num_features=out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bnorm(x)
        return x

class Layer2(torch.nn.Module):
    """
    Input = (64, 56, 56),
    Output = (64, 56, 56)
    """
    def __init__(self):
        super(Layer2, self).__init__()
        self.layer1 = torch.nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Sequential(
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            ConvBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        )
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        identity = x
        x = self.layer1(x)
        x +=  identity
        x = self.relu1(x)
        identity = x
        x = self.layer2(x)
        x += identity
        x = self.relu2(x)
        return x
    
class Layer3(torch.nn.Module):
    """
    Input = (64, 56, 56),
    Output = (128, 28, 28)
    """
    def __init__(self):
        super(Layer3, self).__init__()
        self.projection = ConvBlock(in_channels=64, out_channels=128, kernel_size=1, stride=2)
        self.layer1 = torch.nn.Sequential(
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Sequential(
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            ConvBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        )
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        projection = self.projection(x)
        x = self.layer1(x)
        x += projection
        x = self.relu1(x)
        identity = x
        x = self.layer2(x)
        x += identity
        x = self.relu2(x)
        return x
    
class Layer4(torch.nn.Module):
    """
    Input = (128, 28, 28),
    Output = (256, 14, 14)
    """
    def __init__(self):
        super(Layer4, self).__init__()
        self.projection = ConvBlock(in_channels=128, out_channels=256, kernel_size=1, stride=2)
        self.layer1 = torch.nn.Sequential(
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        )
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Sequential(
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        )
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        projection = self.projection(x)
        x = self.layer1(x)
        x += projection
        x = self.relu1(x)
        identity = x
        x = self.layer2(x)
        x += identity
        x = self.relu2(x)
        return x
    
class Layer5(torch.nn.Module):
    """
    Input = (256, 14, 14),
    Output = (512, 7, 7)
    """
    def __init__(self):
        super(Layer5, self).__init__()
        self.projection = ConvBlock(in_channels=256, out_channels=512, kernel_size=1, stride=2)
        self.layer1 = torch.nn.Sequential(
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        )
        self.relu1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Sequential(
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            ConvBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        )
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        projection = self.projection(x)
        x = self.layer1(x)
        x += projection
        x = self.relu1(x)
        identity = x
        x = self.layer2(x)
        x += identity
        x = self.relu2(x)
        return x
    


  
class ResNet18(torch.nn.Module):
    """
    Input = (3, 224, 224),
    Output = (1000)
    """
    def __init__(self):
        super(ResNet18, self).__init__()
        self.layer1 = torch.nn.Sequential(
            ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer2 = Layer2()
        self.layer3 = Layer3()
        self.layer4 = Layer4()
        self.layer5 = Layer5()
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = torch.nn.Linear(in_features=512, out_features=1000)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x