import torch
import torch.nn as nn

from models.model_base import InitializationMixin


class SENet(nn.Module, InitializationMixin):
    """
    SENet from arXiv:1709.01507 Squeeze-and-Excitation Networks

    Notes
    -----
    Related papers
    Spatial SE arXiv:1803.02579 Concurrent Spatial and Channel Squeeze & Excitation in
    Fully Convolutional Networks
    """
    def __init__(self, classes, img_channels, model_depth, arch_config=None,
            pretrained=False):
        super().__init__()
        pass

    def forward(self, x):
        pass


class SEBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, act=nn.ReLU(inplace=True),
            sigmoid=nn.Sigmoid(), **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = in_channels
        self.block = self.__configure(act, sigmoid)

    def forward(self, x):
        out = self.block(x)
        return out * x

    def __configure(self, act, sigmoid):
        # "Removing the biases of the FC layers in the excitation operation
        # facilitates the modelling of channel dependencies..." p.8
        se_block = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.in_channels, self.mid_channels, 1, bias=True),
                act,
                nn.Conv2d(self.mid_channels, self.out_channels, 1, bias=True),
                sigmoid,
        )
        return se_block


if __name__ == '__main__':
    torch.manual_seed(0)
    se_block = SEBlock(64, 64 // 16, act=torch.nn.ReLU6(inplace=True))

    x = torch.ones(1, 64, 32, 32)
    out = se_block(x)
    print(out)
    print(out.shape)

