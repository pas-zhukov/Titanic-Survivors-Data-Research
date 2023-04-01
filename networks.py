import torch  # библиотека pytorch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.layer_1 = nn.Linear(16, 512)
        self.layer_2 = nn.Linear(512, 2048)
        self.layer_3 = nn.Linear(2048, 512)
        self.layer_4 = nn.Linear(512, 2)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        x = F.softmax(x)

        return x

if __name__ == '__main__':
    model = Net()
    print(model)