import configparser
import os
import datetime
import torch
import torch.optim as optim
import torch.nn as nn
from networks import Net
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import neptune
from data_functions import TitanicDataset, get_loaders
from metrics_functions import compute_binary_accuracy, validation_loss

if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)

model = Net()
model.to(device)
model.type(torch.cuda.FloatTensor)

# Hyper Params
num_epochs = 2000
batch_size = 64
learning_rate = 1e-2
weight_decay = 1e-2
validation_split = .1

# Загрузка данных
data_train = TitanicDataset(normalize=True)
train_loader, val_loader = get_loaders(batch_size=batch_size, data_train=data_train, validation_split=validation_split)

# Loss Function, Optimizer
loss = nn.CrossEntropyLoss().type(torch.cuda.FloatTensor)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# LR Annealing
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=.6, patience=100)

config = configparser.ConfigParser()
config.read('config.ini')

run = neptune.init_run(
    project="pas-zhukov/Titanic-Kaggle",
    api_token=config['Config']['api_token'],
    source_files=['networks.py', 'learning.ipynb', 'metrics_functions.py', 'data_functions.py', 'learning.py']
)
params = {
    'num_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'weight_decay': weight_decay,
    'validation_split': validation_split,
    'optimizer': 'Adam',
    'annealing_factor': .6
}
run["parameters"] = params

loss_history = []
val_loss_history = []
train_history = []
val_history = []
lr_history = []

for epoch in tqdm(range(num_epochs)):
    model.train()

    loss_accum = 0
    correct_samples = 0
    total_samples = 0
    for i_step, (x, y) in enumerate(train_loader):
        #run['train/batch/lr'].append(scheduler.optimizer.param_groups[0]['lr'])
        x = x.to(device)
        y = y.to(device)

        prediction = model(x)
        loss_value = loss(prediction, y)
        #run['train/batch/loss'].append(loss)
        optimizer.zero_grad()
        loss_value.backward()
        # Обновляем веса
        optimizer.step()

        # Определяем индексы, соответствующие выбранным моделью лейблам
        _, indices = torch.max(prediction, dim=1)
        # Сравниваем с ground truth, сохраняем количество правильных ответов
        correct_samples += torch.sum(indices == y)
        # Сохраняем количество всех предсказаний
        total_samples += y.shape[0]
        #run['train/batch/acc'].append(correct_samples / total_samples)

        loss_accum += loss_value

    # Среднее значение функции потерь за эпоху
    ave_loss = loss_accum / (i_step + 1)
    # Рассчитываем точность тренировочных данных на эпохе
    train_accuracy = float(correct_samples) / total_samples
    # Рассчитываем точность на валидационной выборке (вообще после этого надо бы гиперпараметры поподбирать...)
    val_accuracy = compute_binary_accuracy(model, val_loader)

    # Сохраняем значения ф-ии потерь и точности для последующего анализа и построения графиков
    loss_history.append(float(ave_loss))
    train_history.append(train_accuracy)
    val_history.append(val_accuracy)

    # Посчитаем лосс на валидационной выборке
    val_loss = validation_loss(model, val_loader, loss)
    val_loss_history.append(val_loss)

    run['train/epoch/loss'].append(ave_loss)
    run['valid/epoch/loss'].append(val_loss)
    run['train/epoch/acc'].append(train_accuracy)
    run['valid/epoch/acc'].append(val_accuracy)
    run['train/epoch/lr'].append(scheduler.optimizer.param_groups[0]['lr'])

    lr_history.append(scheduler.optimizer.param_groups[0]['lr'])
    # Уменьшаем лернинг рейт (annealing)
    scheduler.step(val_loss)

print("Average loss: %f, Train accuracy: %f, Val accuracy: %f" % (ave_loss, train_accuracy, val_accuracy))

test_set = TitanicDataset(test=True, normalize=True)
test_loader = DataLoader(test_set, batch_size=1)

predictions = []
labels = {}

for i_step, (x, y) in enumerate(test_loader):
    x = x.to(device)
    model.eval()
    prediction = model(x)
    predictions.append(torch.argmax(prediction, dim=1))
    labels[i_step + 892] = int(torch.argmax(prediction, dim=1))

output = pd.DataFrame(labels.items(), columns=['PassengerId', 'Survived'])
output.to_csv(os.path.join('outputs/output' + datetime.datetime.now().strftime('%d%m%y%H%M') + '.csv'), index=False)
