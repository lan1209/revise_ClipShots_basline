import torch
import torch.nn as nn
import torch.nn.functional as F


class deepSBD(nn.Module):
    def __init__(self):
        super(deepSBD, self).__init__()
        self.conv1 = nn.Conv3d(3, 96, kernel_size=3, stride=(1, 2, 2), padding=(0, 0, 0), bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.lrn1 = nn.LocalResponseNorm(5, 0.0001, 0.75)
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0)

        self.conv2 = nn.Conv3d(96, 256, kernel_size=3, padding=(1, 2, 2), bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.lrn2 = nn.LocalResponseNorm(5, 0.0001, 0.75)
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0)

        self.conv3 = nn.Conv3d(256, 384, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.conv4 = nn.Conv3d(384, 384, kernel_size=3, stride=1, padding=(1, 1, 1), bias=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv3d(384, 256, kernel_size=3, stride=1, padding=(1, 1, 1), bias=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(100352, 2048)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5, inplace=False)

        self.fc7 = nn.Linear(2048, 2048)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5, inplace=False)

        self.fc8 = nn.Linear(2048, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.relu4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)

        x = x.view(x.size(0), -1)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout1(x)

        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout2(x)

        x = self.fc8(x)

        x = self.softmax(x)
        return x
