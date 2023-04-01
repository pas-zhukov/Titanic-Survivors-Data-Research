import re
import nltk
from collections import Counter
import numpy as np
import pandas as pd
from missingno import matrix
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy


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
    def __init__(self, test=False, normalize=False):
        self.train_data = pd.read_csv(os.path.join('data/clear_train.csv'))
        self.test_data = pd.read_csv(os.path.join('data/clear_test.csv'))

        self.X_train = np.array(self.train_data.loc[:, :'Title_Mrs.'].drop(columns='PassengerId'))
        self.y_train = np.array(self.train_data.loc[:, 'Survived'])

        self.X_test = np.array(self.test_data.loc[:, :'Title_Mrs.'].drop(columns='PassengerId'))
        self.y_test = np.array(self.test_data.loc[:, 'Sex']) # ЗАГЛУШКА!

        self.normalize = normalize
        if self.normalize:
            self.X_train, self.X_test = self.normalize_data(self.X_train, self.X_test)

        self.test = test

    def __len__(self):
        if self.test:
            return self.X_test.shape[0]
        else:
            return self.X_train.shape[0]

    def __getitem__(self, idx):
        if self.test:
            X = torch.Tensor(self.X_test[idx, :])
            y = self.y_test[idx]
            return X, y
        else:
            X = torch.Tensor(self.X_train[idx, :])
            y = self.y_train[idx]
            return X, y


    @staticmethod
    def normalize_data(train_vector: np.ndarray, test_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:

        train_arr, test_arr = train_vector, test_vector
        united_arr = np.concatenate((train_arr, test_arr))

        mean = np.mean(united_arr, axis=0)
        std_deviation = np.std(united_arr, axis=0)

        train_X = (train_arr - mean) / std_deviation
        test_X = (test_arr - mean) / std_deviation

        return train_X, test_X

if __name__ == '__main__':
    ...