import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
from tqdm import tqdm
from torchvision import transforms

from data_functions import *
from metrics_functions import *

# Подключим видеокарту!
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)

# Загрузка датасета
data_train = TitanicDataset('train.csv', test=False)

# Лоадеры для тренировки и валидации
train_loader, val_loader = get_loaders(batch_size=51, data_train=data_train, validation_split=.1)

nn_model = nn.Sequential(
    nn.Linear(181, 181),
    nn.LeakyReLU(inplace=True),
    nn.BatchNorm1d(181),
    nn.Linear(181, 500),
    nn.ELU(inplace=True),
    nn.Linear(500, 2),
    nn.Softmax()
)
nn_model.type(torch.cuda.FloatTensor)
nn_model.to(device)

# Определяем функцию потерь и выбираем оптимизатор
loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
optimizer = optim.Adam(nn_model.parameters(), lr=1e-4, weight_decay=1e-1)

# Будем также использовать LR Annealing
scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda ep: 0.9, verbose=False)

# Лучше будем снижать LR на плато !UPDATE - не будем :)
'''scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.1, patience=5)'''

# Создадим списки для сохранения величины функции потерь, точности на тренировки и валидации - на каждом этапе (эпохе)
loss_history = []
val_loss_history = []
train_history = []
val_history = []

# Запускаем тренировку!
num_epochs = 200
for epoch in tqdm(range(num_epochs)):
    nn_model.train()  # Enter train mode

    loss_accum = 0
    correct_samples = 0
    total_samples = 0
    for i_step, (x, y) in enumerate(train_loader):
        # Сохраняем наши тензоры в памяти видеокарты, чтобы всё посчитать побыстрее
        x = x.to(device)
        y = y.to(device)

        # Получаем предсказание с существующими весами
        prediction = nn_model(x)
        # Считаем величину функции потерь
        loss_value = loss(prediction, y)
        # Очищаем градиент
        optimizer.zero_grad()
        # Считаем свежий градиент обратным проходом
        loss_value.backward()
        # Обновляем веса
        optimizer.step()

        # Определяем индексы, соответствующие выбранным моделью лейблам
        _, indices = torch.max(prediction, dim=1)
        # Сравниваем с ground truth, сохраняем количество правильных ответов
        correct_samples += torch.sum(indices == y)
        # Сохраняем количество всех предсказаний
        total_samples += y.shape[0]

        # Аккумулируем значение функции потерь, это пригодится далее
        loss_accum += loss_value

    # Среднее значение функции потерь за эпоху
    ave_loss = loss_accum / (i_step + 1)
    # Рассчитываем точность тренировочных данных на эпохе
    train_accuracy = float(correct_samples) / total_samples
    # Рассчитываем точность на валидационной выборке (вообще после этого надо бы гиперпараметры поподбирать...)
    val_accuracy = compute_binary_accuracy(nn_model, val_loader)

    # Сохраняем значения ф-ии потерь и точности для последующего анализа и построения графиков
    loss_history.append(float(ave_loss))
    train_history.append(train_accuracy)
    val_history.append(val_accuracy)

    #Посчитаем лосс на валидационной выборке
    val_loss = validation_loss(nn_model, val_loader, loss)
    val_loss_history.append(val_loss)

    # Уменьшаем лернинг рейт (annealing)
    scheduler.step(val_loss)

    #print(f'Epoch %i of %i' % (epoch, num_epochs))
    #print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))