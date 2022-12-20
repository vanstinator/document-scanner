import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


import torchvision.transforms as torchvision_T


def train_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.RandomGrayscale(p=0.4),
            torchvision_T.Normalize(mean, std),
        ]
    )

    return transforms


def common_transforms(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
    transforms = torchvision_T.Compose(
        [
            torchvision_T.ToTensor(),
            torchvision_T.Normalize(mean, std),
        ]
    )

    return transforms


class SegDataset(Dataset):
    def __init__(
        self, *, img_paths, mask_paths, image_size=(384, 384), data_type="train"
    ):
        self.data_type = data_type
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.image_size = image_size

        if self.data_type == "train":
            self.transforms = train_transforms()
        else:
            self.transforms = common_transforms()

    def read_file(self, path):
        file = cv2.imread(path)[:, :, ::-1]
        file = cv2.resize(file, self.image_size, interpolation=cv2.INTER_NEAREST)
        return file

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):

        image_path = self.img_paths[index]
        image = self.read_file(image_path)
        image = self.transforms(image)

        mask_path = self.mask_paths[index]

        gt_mask = self.read_file(mask_path).astype(np.int32)

        _mask = np.zeros((*self.image_size, 2), dtype=np.float32)

        # BACKGROUND
        _mask[:, :, 0] = np.where(gt_mask[:, :, 0] == 0, 1.0, 0.0)
        # DOCUMENT
        _mask[:, :, 1] = np.where(gt_mask[:, :, 0] == 255, 1.0, 0.0)

        mask = torch.from_numpy(_mask).permute(2, 0, 1)

        return image, mask
