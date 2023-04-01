import re
import nltk
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.feature_extraction.text import CountVectorizer
import os

# Подключим видеокарту!
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)


def get_loaders(batch_size, data_train, validation_split=.2):

    # Определяем количество примеров в фолде валидации
    split = int(np.floor(validation_split * len(data_train)))

    # Список индексов для тренировочных примеров
    indices = list(range(len(data_train)))

    # Рандомизируем положение индексов в списке
    np.random.shuffle(indices)

    # Определяем списки с индексами примеров для тренировки и для валидации
    train_indices, val_indices = indices[split:], indices[:split]

    # Создаем семплеры, которые будут случайно извлекать данные из набора данных
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # Создаем объекты типа ДатаЛоадер, которые будут передавать батчами данные в модель
    train_loader = DataLoader(data_train, batch_size=batch_size,
                              sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(data_train, batch_size=batch_size,
                            sampler=val_sampler, num_workers=4)

    return train_loader, val_loader


class TitanicDataset(Dataset):
    def __init__(self, test=False):
        self.train_data = pd.read_csv(os.path.join('data/clear_train.csv'))
        self.test_data = pd.read_csv(os.path.join('data/clear_test.csv'))

        self.X_train = self.train_data.loc[:, :'Title_Mrs.'].drop(columns='PassengerId').to_numpy(dtype=np.float64)
        self.y_train = self.train_data.loc[:, 'Survived'].to_numpy(dtype=np.float64)

        self.X_test = self.test_data.loc[:, :'Title_Mrs.'].drop(columns='PassengerId').to_numpy(dtype=np.float64)
        self.y_test = self.train_data.loc[:, 'Sex'].to_numpy(dtype=np.float64) # ЗАГЛУШКА!

        self.test = test

    def __len__(self):
        if self.test:
            return self.X_test.shape[0]
        else:
            return self.X_train.shape[0]

    def __getitem__(self, idx):
        if self.test:
            X = torch.Tensor(self.X_test[idx, :])
            y = torch.Tensor(self.y_test[idx])
            return X, y
        else:
            X = torch.Tensor(self.X_train[idx, :])
            y = torch.Tensor(self.y_train[idx])
            return X, y
