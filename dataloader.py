import os

import torchvision
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from time import time
class CustomDataset(Dataset):
    def __init__(self, root_dir, resize_factor):

        data_augmentation = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(resize_factor, interpolation=InterpolationMode.BILINEAR),
                torchvision.transforms.RandomHorizontalFlip(p=0.5),
                # torchvision.transforms.GaussianBlur((51,51), sigma=(0.1, 2.0)),
                # torchvision.transforms.Normalize((0.3, 0.3, 0.3), (0.1, 0.1, 0.1))

            ])

        dataset = torchvision.datasets.ImageFolder(root_dir, transform=data_augmentation)

        self.train_dataset = []
        self.val_dataset = []

        # 일단 순서대로 저장이 돼 있을텐데 하나는 txt파일에 따라 순서대로 traindatset testdataset에 사진과 index를 넣어줄 것이다.
        line = read_txt('./CUB_200_2011/train_test_split.txt')
        print('[Training_data]')
        self.train_dataset = [dataset[i] for i in tqdm(range(len(line))) if line[i][-1] == '1']
        print()
        print('[Validating_data]')
        self.val_dataset = [dataset[i] for i in tqdm(range(len(line))) if line[i][-1] == '0']

    def __len__(self):
        return len(self.train_dataset), len(self.val_dataset)

    def __getitem__(self, idx):
        return self.train_dataset[idx], self.val_dataset[idx]


def read_txt(root_dir):
    f = open(root_dir, mode='r')
    line = f.read().split('\n')
    del line[-1]
    line = sorted(line, key=lambda x: int(x.split(' ')[0]))
    return line


def Dataloader(dataset, batch_size):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    return dataloader

