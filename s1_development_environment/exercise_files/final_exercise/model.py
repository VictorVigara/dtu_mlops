from torch import nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_features, n_hidden, n_classes):
        super().__init__()

        self.fc1 = nn.Linear(n_features, n_hidden[0])
        self.fc2 = nn.Linear(n_hidden[0], n_hidden[1])
        self.fc3 = nn.Linear(n_hidden[1], n_hidden[2])
        self.fc4 = nn.Linear(n_hidden[2], n_classes)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)
        return x

if __name__ == '__main__': 
    model = Net(784, [100, 100, 100], 10)

    a = torch.rand((64, 1, 28, 28))

    output = model(a)
    print(output)