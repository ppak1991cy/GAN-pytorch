import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


image_type = [".jpg", ".png"]


def get_file_list(path, ends=None):
    """ Get all files path in root path. """

    if ends is not None and type(ends) is not list:
        ends = list(ends)

    files_path = []
    for root, dirs, files in os.walk(path):
        if ends is None:
            for f in files:
                f_path = os.path.join(root, f)
                files_path.append(f_path)
        else:
            for f in files:
                if True in [f.endswith(e) for e in ends]:
                    f_path = os.path.join(root, f)
                    files_path.append(f_path)

    return files_path


def normalize_np_image(matrix):
    """ Scale image matrix from 0~255 to -1~1. """

    matrix_new = matrix.copy()
    matrix_new = matrix_new.transpose(2, 0, 1)
    matrix_std = (matrix_new - 127.5) / 127.5
    return matrix_std


def center_crop_matrix(matrix):
    w, h, c = matrix.shape
    side_len = min(w, h)
    w_cut = int((w - side_len) / 2)
    h_cut = int((h - side_len) / 2)
    matrix_new = matrix[w_cut:w_cut + side_len,
                        h_cut:h_cut + side_len].copy()
    return matrix_new


class GANImageDataset(Dataset):
    """ Load image samples. """

    def __init__(self, path, side_len=128):
        self.images_path = get_file_list(path, image_type)
        self.transform = transforms.Compose([
                        transforms.Resize(side_len),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                        ])

    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = Image.open(image_path).convert('RGB')
        image_np = np.asarray(image)
        image_center_np = center_crop_matrix(image_np)
        image_center = Image.fromarray(image_center_np)
        image_trans = self.transform(image_center)

        return image_trans

    def __len__(self):
        return len(self.images_path)


class DataIterator(object):
    """ The iterator used to batch from DataLoader. """

    def __init__(self, dataset, batch_size=8, shuffle=False,
                 num_workers=1, pin_memory=True, drop_last=False):
        self.dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        self.dataiter = iter(self.dataloader)

    def __iter__(self):
        return self.dataiter

    def __next__(self):
        batch = next(self.dataiter, None)
        if batch is None:
            dataiter = iter(self.dataloader)
            batch = next(dataiter, None)
        return batch

    def next(self):
        return next(self)
