import os
import torch
import torch.nn as nn
import torchvision


class InitializationMixin:
    """
    Mixin for pytorch models that allows pretraining of Imagenet models and initialization
    of parameters.

    methods:
        pretrain_file: loads model from file to state_dict
        pretrain_torchvision: loads torchvision model to self.torchvision_model
        initialize_parameters: recursively initializes parameters
    """
    def pretrain_file(self, pretrained):
        # when given state_dict
        if isinstance(pretrained, dict):
            self.load_state_dict(pretrained)
            print('--> Pretrained from state dict')
        # when given path
        elif isinstance(pretrained, str):
            checkpoint_dict = torch.load(pretrained)
            pretrained_params = checkpoint_dict['model_state_dict']
            self.load_state_dict(pretrained_params)
            print('--> Pretrained from {}'.format(pretrained))

    def pretrain_torchvision(self, model_depth, pretrained):
        torchvision_implementation_dict = {
                'mobilenet_v2': torchvision.models.mobilenet_v2,
                'mobilenet_v3_large': torchvision.models.mobilenet_v3_large,
                'mobilenet_v3_small': torchvision.models.mobilenet_v3_small,
        }
        if isinstance(model_depth, str):
            from_torchvision = any([model_depth == t for t in torchvision_implementation_dict.keys()])
            if any([from_torchvision]) and pretrained == True:
                if from_torchvision:
                    print('--> Pretrained from torchvision')
                    self.torchvision_model = torchvision_implementation_dict[model_depth](pretrained)
            elif pretrained == True:
                print(
                        'No official implementations of {}, continuing without pretraining'.format(model_depth)
                )
            else:
                pass

    def initialize_parameters(self, params, method='kaiming_normal', activation='relu'):
        def recursive_initialization(p, **kwargs):
            if any(hasattr(p, i) for i in ['weight', 'bias']):
                self.__initialize(p, **kwargs)
            elif callable(p.children):
                for m in p.children():
                    recursive_initialization(m, **kwargs)
        recursive_initialization(params, method=method, activation=activation)

    def __initialize(self, params, method, activation):
        initialization_implementation_dict = {
                'normal': nn.init.normal_,
                'xavier_normal': nn.init.xavier_normal_,
                'xavier_uniform': nn.init.kaiming_normal_,
                'kaiming_normal': nn.init.kaiming_normal_,
                'kaiming_uniform': nn.init.kaiming_uniform_,
        }
        initialization_args = {
                'normal': {'mean': 0, 'std': 0.01},
                'xavier': {'gain': nn.init.calculate_gain(activation)},
                'kaiming': {'mode': 'fan_out', 'nonlinearity': activation},
        }
        if isinstance(params, (nn.Conv2d, nn.ConvTranspose2d)):
            initialization_implementation_dict[method](
                    params.weight, **initialization_args[method.split('_')[0]])
        # 1706.02677 Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
        # zeroing weights in last bn in res/bottleneck blocks improves 0.2~0.3%
        elif isinstance(params, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm2d)):
            nn.init.constant_(params.weight, 1)
        elif isinstance(params, nn.Linear):
            initialization_implementation_dict['normal'](
                    params.weight, **initialization_args['normal'])
        # zero all biases
        # 1812.01187 Bag of Tricks for Image Classification with Convolutional Neural
        # Networks
        # in regard to wd,
        # "biases and gamma and beta in BN layers, are left unregularized" p.3
        if params.bias is not None:
            nn.init.constant_(params.bias, 0)


# saves model and other variables as checkpoint
def save_checkpoint(path, model: nn.Module, **kwargs):
    # save in first rank when distributed to reduce write overhead
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {}
    if torch.cuda.device_count() > 1:
        save_dict['model_state_dict'] = model.module.state_dict()
    else:
        save_dict['model_state_dict'] = model.state_dict()

    for key in kwargs.keys():
        if key == 'optimizer':
            save_dict['optimizer_state_dict'] = kwargs[key].state_dict()
        elif key == 'scheduler':
            save_dict['scheduler_state_dict'] = kwargs[key].state_dict()
        else:
            save_dict[key] = kwargs[key]
    torch.save(save_dict, path)


# loads model and other variables as dict
def load_checkpoint(path) -> dict:
    return torch.load(path)


