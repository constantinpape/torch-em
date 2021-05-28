import os
import imageio
import h5py
import numpy as np


def create_segmentation_test_data(data_path, raw_key, label_key, shape, chunks):
    with h5py.File(data_path, 'a') as f:
        try:
            f.create_dataset(raw_key, data=np.random.rand(*shape), chunks=chunks)
        except ValueError:  # Unable to create dataset (name already exists)
            pass

        try:
            f.create_dataset(label_key, data=np.random.randint(0, 4, size=shape), chunks=chunks)
        except ValueError:  # Unable to create dataset (name already exists)
            pass


def create_image_collection_test_data(folder, n_images, min_shape, max_shape):
    im_folder = os.path.join(folder, 'images')
    label_folder = os.path.join(folder, 'labels')
    os.makedirs(im_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for i in range(n_images):
        shape = tuple(np.random.randint(mins, maxs) for mins, maxs in zip(min_shape, max_shape))
        raw = np.random.rand(*shape).astype('int16')
        label = np.random.randint(0, 4, size=shape)
        imageio.imwrite(os.path.join(im_folder, f"im_{i}.tif"), raw)
        imageio.imwrite(os.path.join(label_folder, f"im_{i}.tif"), label)
