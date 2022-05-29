import torchvision.transforms as transforms
import numpy as np
import torch
import csv
from torch.utils.data import Dataset
from typing import List

array_alphabet = list(range(25))
array_alphabet.pop(9)


class SignLangMNIST(Dataset):
    @staticmethod
    def __read_label_samples_from_csv(path: str):
        mapping = array_alphabet
        labels, samples = [], []
        with open(path) as file:
            _ = next(file)
            for line in csv.reader(file):
                label = int(line[0])
                labels.append(mapping.index(label))
                samples.append(list(map(int, line[1:])))
        return labels, samples

    def __init__(self, path: str, mean: List[float] = [0.485], std: List[float] = [0.229]):
        labels, samples = SignLangMNIST.__read_label_samples_from_csv(path)
        self._samples = np.array(samples, dtype=np.uint8).reshape((-1, 28, 28, 1))
        self._labels = np.array(labels, dtype=np.uint8).reshape((-1, 1))

        self._mean = mean
        self._std = std

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(28, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self._mean, std=self._std)])

        return {
            'image': transform(self._samples[idx]).float(),
            'label': torch.from_numpy(self._labels[idx]).float()
        }


def get_train_test_loaders(batch_size=32):
    train_set = SignLangMNIST('../resources/data/sign_mnist_train.csv')
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = SignLangMNIST('../resources/data/sign_mnist_test.csv')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader