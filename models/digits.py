import torch
import torch.nn as nn


IN_CHANNELS = 1


def set_in_channels(n):
    global IN_CHANNELS
    IN_CHANNELS = int(n)
    return IN_CHANNELS


def get_in_channels():
    return IN_CHANNELS


class LeNet(nn.Sequential):
    def __init__(self, num_classes=9, in_ch=None):
        in_ch = IN_CHANNELS if in_ch is None else int(in_ch)
        super(LeNet, self).__init__(
            nn.Conv1d(in_ch, 32, kernel_size=64),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=16),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten(start_dim=1),

        )
        self.num_classes = num_classes
        self.out_features = 256

    def copy_head(self):
        return nn.Linear(256, self.num_classes)


def lenet(pretrained=False, **kwargs):
    return LeNet(**kwargs)


class DTN(nn.Sequential):
    def __init__(self, num_classes=10):
        super(DTN, self).__init__(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            nn.Flatten(start_dim=1),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(),
        )
        self.num_classes = num_classes
        self.out_features = 512

    def copy_head(self):
        return nn.Linear(512, self.num_classes)


def dtn(pretrained=False, **kwargs):
    return DTN(**kwargs)


class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activation = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, l = x.size()

        n = l - 1

        x_minus_mu_square = (x - x.mean(dim=2, keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=2, keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activation(y)


class SimCNN(nn.Sequential):
    def __init__(self, num_classes=10, simam=True, in_ch=None):
        in_ch = IN_CHANNELS if in_ch is None else int(in_ch)
        super(SimCNN, self).__init__(
            nn.Conv1d(in_ch, out_channels=32, kernel_size=64),
            nn.BatchNorm1d(32),
            SimAM(),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=16),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5),
            nn.BatchNorm1d(64),
            SimAM(),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5),
            nn.BatchNorm1d(64),
            SimAM(),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten(start_dim=1),
        )
        self.num_classes = num_classes
        self.out_features = 256

    def copy_head(self):
        return nn.Linear(256, self.num_classes)

def simcnn(pretrained=False, **kwargs):
    return SimCNN(**kwargs)
