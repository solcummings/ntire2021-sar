import math
import torch
import torch.nn as nn

from models.model_base import InitializationMixin
from models import senet


class MobileNet(nn.Module, InitializationMixin):
    """
    MobileNet v2 and v3

    MobileNet v2 from arXiv:1801.04381 MobileNetV2: Inverted Residuals and Linear
    Bottlenecks

    MobileNet v3 from arXiv:1905.02244 Searching for MobileNetV3

    Parameters
    ----------
    classes: int
        Number of classes to predict for.
    img_channels: int
        Number of input image channels.
    version: str or list
        Name or list for the number of blocks.
        Names = [v2, v3_large, v3_small]
        List must be [[kernel, mid_channels, out_channels, se, activation, stride]]
    arch_config: dict
        Further configurations for blocks.
        Default = {width: 1., stem_downsample: True, norm: nn.BatchNorm2d}
    pretrained: bool or str
        True pretrains using torchvision.models.mobilenet_${version}
        Path loads the state_dict from a file.

    Notes
    -----
    MobileNet v1 arXiv:1704.04861 MobileNets: Efficient Convolutional Neural Networks for
    Mobile Vision Applications
    MobileNet v2 arXiv:1801.04381 MobileNetV2: Inverted Residuals and Linear Bottlenecks.
    MobileNet v3 arXiv:1905.02244 Searching for MobileNetV3
    """
    def __init__(self, classes, img_channels, version, arch_config=None,
            pretrained=False):
        super().__init__()
        self.activation_dict = {
                'relu': nn.ReLU(inplace=True),
                'relu6': nn.ReLU6(inplace=True),
                'hswish': nn.Hardswish(inplace=True),
        }
        self.version_dict = {
                'v2': [
                    # kernel, mid_channels, out_channels, se, activation, stride
                    # [stem],
                    [3, 32, 16, False, 'relu6', 1],
                    [3, 96, 24, False, 'relu6', 2],
                    [3, 144, 24, False, 'relu6', 1],
                    [3, 144, 32, False, 'relu6', 2],
                    [3, 192, 32, False, 'relu6', 1],
                    [3, 192, 32, False, 'relu6', 1],
                    [3, 192, 64, False, 'relu6', 2],
                    [3, 384, 64, False, 'relu6', 1],
                    [3, 384, 64, False, 'relu6', 1],
                    [3, 384, 64, False, 'relu6', 1],
                    [3, 384, 96, False, 'relu6', 1],
                    [3, 576, 96, False, 'relu6', 1],
                    [3, 576, 96, False, 'relu6', 1],
                    [3, 576, 160, False, 'relu6', 2],
                    [3, 960, 160, False, 'relu6', 1],
                    [3, 960, 160, False, 'relu6', 1],
                    [3, 960, 320, False, 'relu6', 1]
                ],
                'v3_large': [
                    # Table 1 in arXiv:1905.02244 Searching for MobileNetV3
                    # torchvision uses relu instead of relu6 but parameters match
                    # kernel, mid_channels, out_channels, se, activation, stride
                    # [stem],
                    [3, 16, 16, False, 'relu', 1],
                    [3, 64, 24, False, 'relu', 2],
                    [3, 72, 24, False, 'relu', 1],
                    [5, 72, 40, True, 'relu', 2],
                    [5, 120, 40, True, 'relu', 1],
                    [5, 120, 40, True, 'relu', 1],
                    [3, 240, 80, False, 'hswish', 2],
                    [3, 200, 80, False, 'hswish', 1],
                    [3, 184, 80, False, 'hswish', 1],
                    [3, 184, 80, False, 'hswish', 1],
                    [3, 480, 112, True, 'hswish', 1],
                    [3, 672, 112, True, 'hswish', 1],
                    [5, 672, 160, True, 'hswish', 2],
                    [5, 960, 160, True, 'hswish', 1],
                    [5, 960, 160, True, 'hswish', 1],
                ],
                'v3_small': [
                    # Table 2 in arXiv:1905.02244 Searching for MobileNetV3
                    # torchvision uses relu, parameters do not match when using relu6
                    # kernel, mid_channels, out_channels, se, activation, stride
                    # [stem],
                    [3, 16, 16, True, 'relu', 2],
                    [3, 72, 24, False, 'relu', 2],
                    [3, 88, 24, False, 'relu', 1],
                    [5, 96, 40, True, 'hswish', 2],
                    [5, 240, 40, True, 'hswish', 1],
                    [5, 240, 40, True, 'hswish', 1],
                    [5, 120, 48, True, 'hswish', 1],
                    [5, 144, 48, True, 'hswish', 1],
                    [5, 288, 96, True, 'hswish', 2],
                    [5, 576, 96, True, 'hswish', 1],
                    [5, 576, 96, True, 'hswish', 1]
                ],
        }
        self.model_depth_dict = {
                'v2': [1, 2, 3, 7, 4],
                'v3_large': [1, 2, 3, 6, 3],
                'v3_small': [1, 2, 5, 3],
        }
        self.classes = classes
        self.img_channels = img_channels
        self.version = version.lower() if isinstance(version, str) else version
        default_arch_config = {
                'width': 1.,
                'stem_downsample': True,
                'norm': nn.BatchNorm2d,
        }
        # act is used for stem, this is replaced when iterating through instanting layers
        if isinstance(self.version, str):
            if self.version[:2] == 'v2':
                default_arch_config['act'] = self.activation_dict['relu']
                default_arch_config['first_out_channels'] = 32
                default_arch_config['pool_in_channels'] = 1280
                default_arch_config['norm_args'] = {}
            elif self.version[:2] == 'v3':
                default_arch_config['act'] = self.activation_dict['hswish']
                default_arch_config['first_out_channels'] = 16
                default_arch_config['norm_args'] = {'eps': 0.001, 'momentum': 0.01}
                if self.version == 'v3_large':
                    default_arch_config['pool_in_channels'] = 960
                elif self.version == 'v3_small':
                    default_arch_config['pool_in_channels'] = 576
            else:
                raise NotImplementedError
        arch_config = {} if arch_config is None else arch_config
        self.arch_config = default_arch_config | arch_config

        self.assemble()
        self.initialize_parameters(self.encoder)
        self.initialize_parameters(self.pool)
        self.initialize_parameters(self.fc)
        self.torchvision_model = None
        # use non-self version to avoid using variable changed to list
        if isinstance(version, str):
            self.pretrain_torchvision('mobilenet_' + version, pretrained)
        if self.torchvision_model is not None:
            self.__finetune_torchvision()
        self.pretrain_file(pretrained)

    def forward(self, x):
        out = self.encoder(x)
        out = self.pool(out)
        out = self.fc(out)
        return out

    def assemble(self):
        self.__assemble_encoder()
        self.__assemble_pool()
        self.__assemble_fc()

    def __assemble_encoder(self):
        if isinstance(self.version, str):
            version_layers_list = self.version_dict[self.version]
            # assign model_depth
            self.model_depth = self.model_depth_dict[self.version]
        first_out_channels = self.arch_config['first_out_channels']
        encoder_list = []
        encoder_list.append(
                StemBlock(self.img_channels, first_out_channels, **self.arch_config)
        )
        for k, mc, oc, se, act, s in version_layers_list:
            in_channels = encoder_list[-1].out_channels
            mid_channels = self.__scale_width(mc, self.arch_config['width'])
            out_channels = self.__scale_width(oc, self.arch_config['width'])
            # squeeze_chanels uses mid_channels as base in torchvision (in_channels?)
            # ensure divisible by 8
            squeeze_channels = int(math.ceil((mid_channels / 4) / 8) * 8)
            # update act for each layer
            self.arch_config['act'] = self.activation_dict[act]
            encoder_list.append(
                    InvertedResBlock(
                        kernelsize=k,
                        in_channels=in_channels,
                        mid_channels=mid_channels,
                        out_channels=out_channels,
                        stride=s,
                        se=se,
                        squeeze_channels=squeeze_channels,
                        **self.arch_config,
                    )
            )
        # activations default to last invertedresblock
        self.arch_config['act'] = self.activation_dict[act]
        # final conv
        final_out_channels = self.__scale_width(
                self.arch_config['pool_in_channels'],
                self.arch_config['width'],
        )
        encoder_list.append(
                LastConvBlock(
                    in_channels=encoder_list[-1].out_channels,
                    out_channels=final_out_channels,
                    **self.arch_config,
                )
        )
        self.encoder = nn.Sequential(*encoder_list)
        # activations default to last invertedresblock
        self.arch_config['act'] = self.activation_dict[act]

    def __assemble_pool(self):
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.pool[-1].out_channels = self.encoder[-1].out_channels

    def __assemble_fc(self):
        if self.version == 'v2':
            self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(0.2, inplace=True),
                    nn.Linear(self.pool[-1].out_channels, self.classes, bias=True),
            )
        elif self.version[:2] == 'v3':
            if self.version == 'v3_large':
                final_out_channels = 1280
            elif self.version == 'v3_small':
                final_out_channels = 1024
            self.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(self.pool[-1].out_channels, final_out_channels, bias=True),
                    self.arch_config['act'],
                    nn.Dropout(0.2, inplace=True),
                    nn.Linear(final_out_channels, self.classes, bias=True),
            )

    def __finetune_torchvision(self):
        # loads torchvision parameters to self.encoder and self.fc
        pretrained_params = nn.Sequential(*list(self.torchvision_model.children())[0])
        if self.arch_config['width'] != 1.:
            print('--> Not pretraining')
        else:
            if self.img_channels != 3:
                print('--> Not pretraining stem')
                # loads torchvision params except stem
                self.encoder.load_state_dict(
                        {**self.encoder[:1].state_dict(),
                            **dict(zip(self.encoder[1:].state_dict().keys(),
                                pretrained_params[1:].state_dict().values()))}
                )
            else:
                self.encoder.load_state_dict(
                        dict(zip(self.encoder.state_dict().keys(),
                            pretrained_params.state_dict().values()))
                )
            if self.classes == 1000:
                self.fc = nn.Sequential(
                        nn.Flatten(),
                        *list(self.torchvision_model.children())[-1],
                )
        del self.torchvision_model

    @staticmethod
    def __scale_width(value, width_scale, factor=8):
        return int((value * width_scale + factor / 2) // factor * factor)


class StemBlock(nn.Module):
    def __init__(self, in_channels, out_channels=32,
            norm=nn.BatchNorm2d, norm_args={}, act=nn.ReLU6(inplace=True),
            stem_downsample=True, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stem = self.__configure(norm, norm_args, act, stem_downsample)

    def forward(self, x):
        return self.stem(x)

    def __configure(self, norm, norm_args, act, stem_downsample):
        stride = 2 if stem_downsample else 1
        block = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, stride, padding=1,
                    bias=False),
                norm(self.out_channels, **norm_args),
                act,
        )
        return block


class InvertedResBlock(nn.Module):
    def __init__(self, kernelsize, in_channels, mid_channels, out_channels, stride, se,
            squeeze_channels, norm=nn.BatchNorm2d, norm_args={},
            act=nn.ReLU6(inplace=True), se_args={}, drop_connect_ratio=0., **kwargs):
        super().__init__()
        self.kernelsize = kernelsize
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.squeeze_channels = squeeze_channels
        self.stride = stride
        self.padding = (self.kernelsize - 1) // 2

        default_se_args = {'act': nn.ReLU(inplace=True), 'sigmoid': nn.Sigmoid()}
        se_args = default_se_args | se_args

        self.block = self.__configure(norm, norm_args, se, act, se_args)
        self.drop_connect_ratio = drop_connect_ratio
        if self.drop_connect_ratio > 0 and self.stride == 1 \
                and self.in_channels == self.out_channels:
            self.drop_connect = DropConnect(self.drop_connect_ratio)

    def forward(self, x):
        out = self.block(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            if self.drop_connect_ratio > 0:
                out = self.drop_connect(out)
            out += x
        return out

    def __configure(self, norm, norm_args, se, act, se_args):
        if self.in_channels == self.mid_channels:
            pointwise_layer = nn.Sequential()
        else:
            pointwise_layer = nn.Sequential(
                    nn.Conv2d(self.in_channels, self.mid_channels, 1, bias=False),
                    norm(self.mid_channels, **norm_args),
                    act,
            )
        depthwise_layer = nn.Sequential(
                nn.Conv2d(
                    self.mid_channels, self.mid_channels, self.kernelsize, self.stride,
                    padding=self.padding, groups=self.mid_channels, bias=False),
                norm(self.mid_channels, **norm_args),
                act,
        )
        if se:
            # seblock always uses relu and hardsigmoid for mobilenet v3
            se_block = senet.SEBlock(
                    self.mid_channels,
                    self.squeeze_channels,
                    **se_args,
            )
        else:
            se_block = nn.Sequential()
        pointwise_layer_no_act = nn.Sequential(
                nn.Conv2d(self.mid_channels, self.out_channels, 1, bias=False),
                norm(self.out_channels, **norm_args),
        )
        return nn.Sequential(
                *pointwise_layer, *depthwise_layer, se_block, *pointwise_layer_no_act)


class LastConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=1280,
            norm=nn.BatchNorm2d, norm_args={}, act=nn.ReLU6(inplace=True), **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.last_block = self.__configure(norm, norm_args, act)

    def forward(self, x):
        return self.last_block(x)

    def __configure(self, norm, norm_args, act):
        last_block = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
                norm(self.out_channels, **norm_args),
                act,
        )
        return last_block


class DropConnect(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            keep_ratio = 1. - self.p
            mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
            mask.bernoulli_(keep_ratio)
            x.div_(keep_ratio)
            x.mul_(mask)
        return x

    def extra_repr(self) -> str:
        return 'p={}'.format(self.p)


if __name__ == '__main__':
    import torchvision
    torch.manual_seed(0)
    model = MobileNet(1000, 3, 'v3_large', pretrained=True).eval()
    print(model)
    torchvision_model = torchvision.models.mobilenet_v3_large(True).eval()
    x = torch.rand(1, 3, 224, 224)

    model_output = model(x)
    print(model_output.shape)
    torchvision_model_output = torchvision_model(x)

    print(torch.sum(model_output - torchvision_model_output))

