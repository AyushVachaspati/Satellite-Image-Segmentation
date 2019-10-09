"""
Script that caches train data for future training
"""



from __future__ import division
from os import walk
import os
import extra_functions
import h5py
import numpy as np

data_path = 'FP_Mask_3'

def cache_train():

    fp_files = []
    for (dirpath, dirnames, filenames) in walk(data_path):
        fp_files.extend(filenames)
        break

    print("Number of Training images: ", len(fp_files))
    print("Processing...")

    num_channels = 3
    num_mask_channels = 1
    image_rows = 3705
    image_cols = 4800
    num_train = len(fp_files)

    f = h5py.File(os.path.join(data_path, 'train_16.h5'), 'w')

    imgs = f.create_dataset('train', (num_train, image_rows, image_cols,num_channels), dtype=np.float16)
    imgs_mask = f.create_dataset('train_mask', (num_train, image_rows, image_cols,num_mask_channels), dtype=np.uint8)

    ids = []
    i = 0

    for image_id in sorted(fp_files):
        print(image_id)
        image = extra_functions.read_image(image_id)
        mask = extra_functions.read_mask(image_id)
        height, width, _ = image.shape

        imgs[i] = image
        imgs_mask[i] = mask

        ids += [image_id]
        i += 1

    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')

    f.close()
    print("Training Images Cached Successfully.")

if __name__ == '__main__':
    cache_train()
