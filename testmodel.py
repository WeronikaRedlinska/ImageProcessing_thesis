#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from torchvision.transforms import functional as F
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
import time
from eval import _eval
from models.MIMOUNet import build_net
from torch.utils.data import Dataset, DataLoader
from PIL import Image as Image
from data import test_dataloader


# In[2]:


class DeblurDataset(Dataset):
    def __init__(self, image_dir, transform=None, is_test=False):
        self.image_dir = image_dir
        self.image_list = os.listdir(os.path.join(image_dir))
        self._check_image(self.image_list)
        self.image_list.sort()
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_dir,self.image_list[idx]))

        if self.transform:
            image= self.transform(image)
        else:
            image = F.to_tensor(image)
        if self.is_test:
            name = self.image_list[idx]
            return image, name
        return image

    @staticmethod
    def _check_image(lst):
        for x in lst:
            splits = x.split('.')
            if splits[-1] not in ['png', 'jpg', 'jpeg']:
                raise ValueError


# In[3]:


def try_dataloader(path, batch_size=1, num_workers=0):
    dataloader = DataLoader(
        DeblurDataset(path, is_test=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader






