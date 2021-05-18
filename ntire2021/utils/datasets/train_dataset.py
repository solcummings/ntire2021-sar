import logging
logger = logging.getLogger(__name__)
import os
import random
import yaml
import numpy as np
import pandas as pd

from utils.transforms.compose import ComposeTransforms


class Dataset:
    def __init__(self, phase, img_file, label_file, mean_std_file, transforms_args,
            val_csv=None, sampling_method=False, sampling_size=None, val_ratio=0.2,
            seed=0, class_list=[[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]], **kwargs):
        self.phase = phase
        self.img_array = np.load(img_file)
        self.label_array = np.load(label_file)
        self.img_filename = os.path.basename(img_file).split('.')[0]
        self.sampling_method = sampling_method
        self.sampling_size = sampling_size
        self.class_list = class_list

        self.val_ratio = val_ratio
        self.seed = seed

        # split train/val
        self.img_array, self.label_array = self._split_df(val_csv)

        logger.info('{} has {} patches'.format(self.phase.title(), len(self.img_array)))
        self.transforms = ComposeTransforms(
                transforms_dict=transforms_args,
                mean_std=mean_std_file,
        )

    def __getitem__(self, index):
        img = self.img_array[index]
        label = np.array(self.label_array[index])

        if self.transforms.mix:
            mix_index = random.randrange(0, len(self))
            mix_img = self.img_array[mix_index]
            mix_label = np.array(self.label_array[mix_index])
            img, label = self.transforms(
                    x=[img, label],
                    name=self.img_filename,
                    mix_x=[mix_img, mix_label],
            )
        else:
            # transforms return list of array
            img, label = self.transforms([img, label], self.img_filename)

        return img, label

    def _split_df(self, val_csv):
        if val_csv:
            val_df = pd.read_csv(val_csv)
        split_label_list = []
        split_img_list = []
        total_n = len(self.label_array)
        classes = np.unique(self.label_array)
        np.random.seed(self.seed)
        for c, grouped_classes in enumerate(self.class_list):
            for g in grouped_classes:
                class_index = np.where(self.label_array == g)[0]  # np.where returns tuple
                np.random.shuffle(class_index)
                if val_csv:
                    already_val = val_df['npy_index'][val_df['label'] == g].values
                    class_index = np.array([i for i in class_index if i not in
                        already_val])
                    # bring valid first
                    class_index = np.concatenate([already_val, class_index])
                val_index = int(len(class_index) * self.val_ratio)
                if self.phase == 'train':
                    class_index = class_index[val_index:]
                else:
                    class_index = class_index[:val_index]
                if self.sampling_method == 'undersample':
                    class_index = class_index[:self.sampling_size]
                elif self.sampling_method == 'oversample':
                    class_index = np.resize(class_index, self.sampling_size)
                split_label_list.append(np.full(len(class_index), c))
                split_img_list.append(self.img_array[class_index])

        split_label_list = np.concatenate(split_label_list)
        split_img_list = np.concatenate(split_img_list)
        for c in range(len(self.class_list)):
            logging.info('{} patches for class {} in {}'.format(
                len(np.where(split_label_list == c)[0]), c, self.phase))
        # shuffle order of images and labels uniformly
        p = np.random.permutation(len(split_label_list))
        self.img_array = split_img_list[p]
        self.label_array = split_label_list[p]
        return self.img_array, self.label_array

    def __len__(self):
        return len(self.label_array)


if __name__ == '__main__':
    pass



