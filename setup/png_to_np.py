import argparse
import os
import numpy as np
from glob import glob
from PIL import Image
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert png images to single .npy file')
    parser.add_argument('--png_dir', type=str,
            default='../ntire2021/data/source/train_images/')
    args = parser.parse_args()

    img_prefix = 'SAR'
    data_dir = args.png_dir
    out_basepath = '../ntire2021/data/train/'
    img_list = []
    label_list = []
    for cls in range(0, 10):
        for img_file in tqdm(
                sorted(glob(os.path.join(data_dir, str(cls), '{}*'.format(img_prefix))))):
            img = np.array(Image.open(img_file))
            # zero pad to 64x64 pixels
            zeros = np.zeros((64, 64), dtype=np.uint8)
            h, w = img.shape
            start_index = (64 - h) // 2
            zeros[start_index: start_index + h, start_index: start_index + w] = img
            img = zeros
            # insert channel dimension
            img = img[None, :]
            img_list.append(img)

            img = np.array(Image.open(img_file))
            label_list.append(cls)

    np.save(
            os.path.join(out_basepath, '{}_images.npy'.format(img_prefix)),
            np.array(img_list, dtype=np.uint8),
    )
    np.save(
            os.path.join(out_basepath, '{}_labels.npy'.format(img_prefix)),
            np.array(label_list, dtype=np.uint8),
    )

