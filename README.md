# Improving Training Strategies for SAR Image Classification
Solution for the NTIRE 2021 Multi-modal Aerial View Object Classification Challenge - Track 1 (SAR) at CVPRW 2021

## Overview
The objective of the challenge is to classify SAR image chips of vehicles into 10 classes. The MobileNet family of architectures are experimented with due to limitations in VRAM and time constraints.

### Methodology
1. Addressing low resolution inputs  
SAR image chips have variable lengths of 54 to 60 pixels per side; small image sizes limit the amount of information that each image chip contains. Accuracy on small targets decreases when inputs are downsampled early on in networks, thus the stride of the convolutional layer within the stem block is changed from two to one.
2. Addressing limited dataset size  
The dataset consists of samples that are alike, which limit the intraclass variation. To compensate, networks are pretrained on ImageNet. However, the stem block is excluded from pretraining due to the difference in stride and the inputs having a single channel as opposed to three.
A center-preserving Cutmix modification coined **Central Cutmix** is employed alongside traditional flipping and rotations to introduce variation during training.
3. Addressing class imbalance  
The classes within the dataset exhibit a large imbalance, where the majority of samples are from the "sedan" class. To ensure balanced performace, all classes are under-sampled to approximately 1400 samples during training, which is approximately 80% of the number of "box truck" samples.

### Results
|MobileNet|Cutmix|Central cutmix|Stem stride=1|Cosine annealing|Accuracy valid data|Accuracy test data|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|V2|||||15.58||
|V2|x||||16.88||
|V2||x|||18.05||
|V3-Large||x|||18.96||
|V3-Large||x|x||21.56||
|V3-Large||x|x|x|22.34|26.39|

## Usage
### Dependencies
- numpy
- pandas
- pillow
- pytorch
- pyyaml
- torchvision
- tqdm

### Downloading and Preprocessing Data

```
cd ./setup/
bash setup.bash
```
### Training
```
cd ./ntire2021/
python train.py
```
- changing the save_dir path in config_train.yml allows for changing the location of the directory that the newly trained model will be saved to

### Predicting
```
cd ./ntire2021/
python predict.py
```

- changing the save_dir path in config_predict.yml allows for changing the location of the directory that the prediction csv file will be saved to


