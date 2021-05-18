import os
import torch
import yaml
import numpy as np
import pandas as pd

from glob import glob
from PIL import Image

from utils.transforms.compose import ComposeTransforms


class Dataset:
    def __init__(self, phase, img_file, mean_std_file, transforms_args, seed=0, **kwargs):
        self.phase = phase
        prefix = 'SAR'
        self.img_file_list = sorted(
                glob(os.path.join(img_file + '{}*.png'.format(prefix))))

        self.band3 = True if '3band' in kwargs and kwargs['3band'] else False
        # transforms in dataset only do normalization and crop
        crop_args = transforms_args.pop('centercrop')
        self.patchsize = crop_args['patchsize']
        self.transforms = ComposeTransforms(
                transforms_dict={'centercrop': crop_args},
                mean_std=mean_std_file,
                to_tensor=False,
        )

    def __getitem__(self, index):
        img_file = self.img_file_list[index]

        img = np.array(Image.open(img_file))
        if self.band3:
            img = np.dstack([img] * 3)
            img = img.transpose(2,0,1)
        else:
            img = img[None, :]
        # transforms return list of array
        img = self.transforms([img], 'SAR_images')[0]
        return img, os.path.basename(img_file).split('_')[1].split('.')[0]

    def __len__(self):
        return len(self.img_file_list)


if __name__ == '__main__':
    pass

