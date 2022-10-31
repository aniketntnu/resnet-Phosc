import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling
# from pyramidpooling import TemporalPyramidPooling

from timm.models.registry import register_model

__all__ = [
    'PHOSCnet_temporalpooling',
    "Resnet18_temporalpooling",
    "Resnet18_temporalpooling2",
]


class PHOSCnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(64, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), padding='same'),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), stride=2),
            nn.Conv2d(128, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 512, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), padding='same'),
            nn.ReLU(),
        )

        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.phos = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 604),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict:
        x = self.conv(x)
        x = self.temporal_pool(x)

        return {'phos': self.phos(x), 'phoc': self.phoc(x)}


@register_model
def PHOSCnet_temporalpooling(**kwargs):
    return PHOSCnet()

import torch

import torch.nn as nn

from modules.pyramidpooling import TemporalPyramidPooling
import torchvision.models as modelsRes



class ResNet18(nn.Module):
    def __init__(self, n_out=0, in_channels=1, gpp_type='tpp', pooling_levels=3, pool_type='max_pool'):
        super().__init__()
        resnet18 = modelsRes.resnet18(pretrained=True)
        #resnet34 = modelsRes.resnet50(pretrained=True)

        self.resnet18= nn.Sequential(*list(resnet18.children())[:-2])
        #self.fc = nn.Linear(512* 2* 8, n_out)
        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.phos1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc1 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(4096, 604),
        )
        
    def forward(self, input_tensor):

        y = input_tensor

        if y.shape[1] == 1:
            y = y.expand((y.shape[0], 3, *y.shape[2:]))
        x = self.resnet18(y)

        x = self.temporal_pool(x)

        return {'phos': self.phos1(x), 'phoc': self.phoc1(x),"x":x}

@register_model
def Resnet18_temporalpooling(**kwargs):
    return ResNet18()

class ResNet18_2(nn.Module):
    def __init__(self, n_out=0, in_channels=1, gpp_type='tpp', pooling_levels=3, pool_type='max_pool'):
        super().__init__()
        resnet18 = modelsRes.resnet18(pretrained=True)
        #resnet34 = modelsRes.resnet50(pretrained=True)

        self.resnet18= nn.Sequential(*list(resnet18.children())[:-2])
        #self.fc = nn.Linear(512* 2* 8, n_out)
        self.temporal_pool = TemporalPyramidPooling([1, 2, 5])

        self.phos1 = nn.Sequential(

            nn.Linear(4096, 165),
            nn.ReLU()
        )

        self.phoc1 = nn.Sequential(
            nn.Linear(4096, 604),
        )

    def forward(self, input_tensor):

        y = input_tensor

        if y.shape[1] == 1:
            y = y.expand((y.shape[0], 3, *y.shape[2:]))
        x = self.resnet18(y)

        x = self.temporal_pool(x)

        return {'phos': self.phos1(x), 'phoc': self.phoc1(x),"x":x}

@register_model
def Resnet18_temporalpooling2(**kwargs):
    return ResNet18_2()


if __name__ == '__main__':
    from torchsummary import summary

    model = PHOSCnet()

    summary(model, (3, 50, 250))
