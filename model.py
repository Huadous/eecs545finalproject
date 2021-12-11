import torch
from torch import nn
from typing import Union, List, Dict, Any, cast
from torchvision.models import VGG

class SCNN(nn.Module):
    def __init__(self, num_classes, input_channel = 1):
        super().__init__()
        self.block1 = self.conv_block(input_channel, out_dim=128)
        self.block2 = self.conv_block(128, out_dim=128)
        self.block3 = self.conv_block(128, 128)
        self.block4 = self.conv_block(128, 256)
        self.block5 = self.conv_block(256, 256)
        self.block6 = self.conv_block(256, 128)

        self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout2d()])
        self.fc = nn.Sequential(nn.Linear(128, num_classes))
        self.average_pool = nn.AdaptiveAvgPool2d((1, 1))

    def conv_block(self, input_dim, out_dim, kernel_size=3, stride=2, padding=2): 
        return nn.Sequential(
            nn.Conv2d(input_dim, out_dim, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True))

    def forward(self, x):
        z = self.block1(x)
        z = self.block2(z)
        z = self.block3(z)
        z = self.maxpool(z)
        z = self.block4(z)
        z = self.block5(z)
        z = self.block6(z)
        z = self.average_pool(z)
        temp = z.shape[0]
        z = z.view(temp, -1)
        z = self.fc(z)
        return z

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:

    model = VGG(make_layers([64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512,], batch_norm=False), **kwargs)

    return model


