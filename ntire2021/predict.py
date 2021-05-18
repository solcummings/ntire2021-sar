import argparse
import os
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm

from common import load_yaml

from utils import datasets
from utils import misc
from utils import transforms
import models


class Predicting:
    def __init__(self, config_path, disable_tqdm=False):
        self.config = load_yaml.load(config_path)
        self.disable_tqdm = disable_tqdm

        misc.seeds.set_seeds(self.config['seed'], self.config['deterministic'])

        self.amp = self.config['amp']
        self.dataloader = datasets.build('test', self.config['dataset_args'])

        self.tta = True if len(self.config['dataset_args']['transforms_args'].keys()) \
                > 1 else False
        if self.tta:
            self.transforms = transforms.tta.TTA(
                    self.config['dataset_args']['transforms_args'])

        self.device = torch.device('cuda')
        self.model = models.build(**self.config)
        self.model = self.model.to(self.device)
        self.softmax = torch.nn.Softmax(dim=1).to(self.device)

        self.classes = self.config['model_args']['classes']

        self.save_dir = self.config['save_dir']
        os.makedirs(self.save_dir, exist_ok=True)

    def predict(self):
        print('--- Starting Predicting ---')
        self.model.eval()
        with torch.no_grad():
            self.iterate_dataloader(self.dataloader)

    def iterate_dataloader(self, dataloader):
        results_dict = {'image_id': [], 'class_id': []}
        percentages_dict = {'class{}_softmax'.format(k): [] for k in range(self.classes)}
        percentages_dict = {'image_id': []} | percentages_dict

        for batch in tqdm(dataloader, disable=self.disable_tqdm):
            img, name_list = batch
            if self.tta:
                # tta returns list for each pattern of list of tensors
                img = self.transforms(img)
                total_prediction = []
                for augmented_input in img:
                    img = augmented_input[0].to(self.device, dtype=torch.float)
                    prediction = self.softmax(self.__autocast_prediction(img))
                    prediction = prediction.to('cpu').numpy()
                    total_prediction.append(prediction)
                total_prediction = self.transforms.reverse(total_prediction)
                # mean tta outputs
                prediction = np.mean(total_prediction, axis=0)
                percentages_dict['image_id'].extend(name_list)
                for c in range(prediction.shape[1]):
                    percentages_dict['class{}_softmax'.format(c)].extend(
                            list(prediction[:, c]))
                # (batch, classes)
                prediction = np.argmax(prediction, axis=1)
            else:
                img = img.to(self.device, dtype=torch.float)
                prediction = self.softmax(self.__autocast_prediction(img))
                prediction = prediction.to('cpu').numpy()
                prediction = np.argmax(prediction, axis=1)

            results_dict['image_id'].extend(name_list)
            results_dict['class_id'].extend(list(prediction))
        df = pd.DataFrame.from_dict(results_dict)
        df.to_csv(os.path.join(self.save_dir, 'results.csv'), index=False)
        print(df['class_id'].value_counts())
        df = pd.DataFrame.from_dict(percentages_dict)
        df.to_csv(os.path.join(self.save_dir, 'percentages.csv'), index=False)

    def __autocast_prediction(self, img):
        if self.amp:
            with torch.cuda.amp.autocast():
                prediction = self.model(img)
        else:
            prediction = self.model(img)
        return prediction


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', type=str, default='./config_predict.yml')
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()

    model = Predicting(args.config, args.disable_tqdm)
    model.predict()


