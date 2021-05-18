#!/bin/bash


data_dir='../ntire2021/data/source/'
download_address='https://competitions.codalab.org/competitions/28123#participate'

if [ -d $data_dir ]; then
    if [ -d $data_dir'/valid_images/' ] && [ -d $data_dir'/train_images/' ] &&
        [ -d $data_dir'/test_images/' ]; then
        echo "Data directory $data_dir exists."
        download=false
        # convert training png to single .npy file for training phase
        python png_to_np.py --png_dir $data_dir'/train_images/'
    else
        echo "Data directory $data_dir exists, but data is missing."
        echo "Please download the data below into the data directory and rerun $0"
        echo "Data can be found at: $download_address"
        download=true
    fi
else
    echo "Data directory $data_dir does not exist."
    echo "Please download the data below into the data directory and rerun $0"
    echo "Data can be found at: $download_address"
    mkdir --parents $data_dir
    download=true
fi

