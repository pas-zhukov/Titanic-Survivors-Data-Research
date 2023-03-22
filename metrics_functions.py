import torch



# Подключим видеокарту!
if torch.cuda.is_available():
    dev = 'cuda:0'
else:
    dev = 'cpu'
device = torch.device(dev)


def compute_binary_accuracy(model, loader):
    """
    Computes accuracy on the dataset wrapped in a loader

    Returns: accuracy as a float value between 0 and 1
    """
    model.eval()  # Evaluation mode

    total_samples = 0
    true_samples = 0

    for i_step, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        prediction = torch.argmax(model(x), dim=1)
        true_samples += int(loader.batch_size - torch.count_nonzero(prediction - y))
        total_samples += loader.batch_size

    return float(true_samples / total_samples)


def validation_loss(model, loader, loss):
    loss_accum = 0
    for i_step, (x, y) in enumerate(loader):
        # Сохраняем наши тензоры в памяти видеокарты, чтобы всё посчитать побыстрее
        x = x.to(device)
        y = y.to(device)
        model.eval()
        # Получаем предсказание с существующими весами
        prediction = model(x)
        # Считаем величину функции потерь
        loss_value = loss(prediction, y)

        # Аккумулируем значение функции потерь, это пригодится далее
        loss_accum += loss_value

    # Среднее значение функции потерь
    ave_loss = loss_accum / (i_step + 1)

    return ave_loss
