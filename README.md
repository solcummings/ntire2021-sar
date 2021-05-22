# Solution for the NTIRE 2021 Multi-modal Aerial View Object Classification Challenge - Track 1 (SAR) at CVPRW 2021

## Dependencies
- numpy
- pandas
- pillow
- pytorch
- pyyaml
- torchvision
- tqdm

## Downloading and Preprocessing Data

```
cd ./setup/
bash setup.bash
```
## Training
```
cd ./ntire2021/
python train.py
```
- changing the save_dir path in config_train.yml allows for changing the location of the directory that the newly trained model will be saved to

## Predicting
```
cd ./ntire2021/
python predict.py
```

- changing the save_dir path in config_predict.yml allows for changing the location of the directory that the prediction csv file will be saved to


