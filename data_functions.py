import re
import nltk
from collections import Counter
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.feature_extraction.text import CountVectorizer

# Подключим видеокарту!
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)


def get_loaders(batch_size, data_train, validation_split=.2):
    # Количество тренировочных примеров
    data_size = len(data_train)

    # Определяем количество примеров в фолде валидации
    split = int(np.floor(validation_split * data_size))

    # Список индексов для тренировочных примеров
    indices = list(range(data_size))

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
    def __init__(self, data_path, transform=None, target_transform=None, test=False):
        self.train_data = pd.read_csv(data_path)
        self.transform = transform
        self.target_transform = target_transform
        '''
        np.array([id,pclass,*name, sex, age, sibsp, parch, *ticket, ticket_n,fare,*cabin,*embarked])
        '''
        self.test = test
        self.X = np.array([self.train_data.PassengerId,
                      self.train_data.Pclass,
                      *(self.string_data_vectorizer(self.train_data.Name, max_features=80).T),
                      self.sexes(),
                      self.restore_age(),
                      self.train_data.SibSp,
                      self.train_data.Parch,
                      *(self.string_data_vectorizer(self.train_data.Ticket, max_features=10, tickets=True).T),
                      self.ticket_numbers(),
                      self.train_data.Fare,
                      *(self.string_data_vectorizer(self.train_data.Cabin, max_features=80, clear_data=False).T),
                      *(self.string_data_vectorizer(self.train_data.Embarked, max_features=3, clear_data=False,
                                                    embarked=True).T)]).T

        if not self.test:
            self.y = np.array(self.train_data.Survived)




    def __len__(self):
        return self.train_data.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx, :]
        if not self.test:
            y = self.y[idx]
        if self.test:
            y = 0

        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)

        return torch.Tensor(x), y

    def string_data_vectorizer(self, string_data, max_features: int, clear_data=True, tickets=False, embarked=False):
        """

        :param string_data: Список векторизуемых строк
        :param max_features: Длина выходного вектора
        :param clear_data: True, если в списке нет NaNов
        :param tickets: True, если векторизуются номера билетов
        :param embarked: True when vectorizing embarked
        :return: ndArray векторов
        """
        clear_string_data = []
        if not clear_data:
            for i, string_ in enumerate(string_data):
                if isinstance(string_, str):
                    clear_string_data.append(string_)
                else:
                    clear_string_data.append('')
        else:
            clear_string_data = string_data

        # Токенизируем
        united_strings = ' '.join(clear_string_data)
        tokens_list_1 = nltk.word_tokenize(united_strings)
        tokens_list_2 = []

        # Почистим от знаков пунктуации
        if not tickets:
            for token in tokens_list_1:
                if token in ['.', ',', '"', '(', ')', "''", "``"]:
                    continue
                else:
                    tokens_list_2.append(token)
        # А билеты также почистим от номеров
        if tickets:
            for token in tokens_list_1:
                try:
                    int(token)
                except:
                    if token in ['.', ',', '"', '(', ')']:
                        continue
                    else:
                        tokens_list_2.append(token)

        clear_strings = []
        for string__ in string_data:
            if type(string__) is str:
                string___ = nltk.word_tokenize(string__)
            else:
                string___ = ['']
            string_out = []
            for token in string___:
                if token in tokens_list_2:
                    string_out.append(token)
            if embarked:
                if len(string_out) != 0:
                    string_out.append('nn')
                clear_strings.append(''.join(string_out))
            if not embarked:
                clear_strings.append(' '.join(string_out))



        # Векторизуем
        counter = CountVectorizer(max_features=max_features)
        vectors = pd.DataFrame(counter.fit_transform(clear_strings).toarray(),
                               index=self.train_data.PassengerId,
                               columns=sorted(counter.vocabulary_.keys())
                               )

        return np.array(vectors)

    def restore_age(self):
        """

        :return: ndArray возрастов с заполненными пропусками
        """
        ids_with_age = []
        ids_no_age = []
        for i, id in enumerate(self.train_data.PassengerId):
            if self.train_data.Age[i] > 0:
                ids_with_age.append(id)
            else:
                ids_no_age.append(id)

        def find_mean_age(sibsp, parch):
            accum_age = 0
            count = 0
            for i, pasid in enumerate(self.train_data.PassengerId):
                if self.train_data.SibSp[i] == sibsp and self.train_data.Parch[i] == parch:
                    if self.train_data.Age[i] > 0:
                        accum_age += self.train_data.Age[i]
                    count += 1

            if accum_age == 0:
                for i, pasid in enumerate(self.train_data.PassengerId):
                    if self.train_data.SibSp[i] == sibsp or self.train_data.Parch[i] == parch:
                        if self.train_data.Age[i] > 0:
                            accum_age += self.train_data.Age[i]
                        count += 1

            return round(accum_age / count, 1)

        restored_ages = {}

        for i, id in enumerate(ids_no_age):
            if self.test:
                index = id-892
            else:
                index = i
            restored_ages[id] = find_mean_age(self.train_data.SibSp[index], self.train_data.Parch[index])

        restored_ages_vector = self.train_data.Age
        for pasid in self.train_data.PassengerId:
            if pasid in restored_ages.keys():
                if self.test:
                    index = pasid - 892
                else:
                    index = pasid - 1
                restored_ages_vector[index] = restored_ages[pasid]

        #print(restored_ages_vector, np.array(restored_ages_vector).shape)
        return np.array(restored_ages_vector)

    def sexes(self):
        sexes = []
        for sex in self.train_data.Sex:
            if sex == 'male': sexes.append(1)
            else: sexes.append(0)
        return np.array(sexes)

    def ticket_numbers(self):
        ticket_numbers = []
        tickets_accum = 0
        count = 0
        for ticket in self.train_data.Ticket:
            try:
                ticket_numbers.append(int(re.split(' ', ticket)[-1]))
                tickets_accum += int(re.split(' ', ticket)[-1])
                count += 1
            except ValueError:
                ticket_numbers.append(int(tickets_accum/count))
                tickets_accum += int(tickets_accum/count)
                count += 1

        return np.array(ticket_numbers)
