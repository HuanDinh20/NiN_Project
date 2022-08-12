import torch
from torch import nn


class NiN(nn.Module):

    def __init__(self, num_classes=10):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(out_channel=96, kernel_size=11, stride=4, padding=0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(out_channel=256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(out_channel=384, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(num_classes, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )

    def forward(self, X):
        X = self.net(X)
        return X

    @staticmethod
    def nin_block(out_channel, kernel_size, stride, padding):
        nin = nn.Sequential(
            nn.LazyConv2d(out_channel, kernel_size=kernel_size, stride=stride, padding=padding), nn.ReLU(),
            nn.LazyConv2d(out_channel, kernel_size=1), nn.ReLU(),
            nn.LazyConv2d(out_channel, kernel_size=1), nn.ReLU())
        return nin

    @staticmethod
    def xavier_uniform(m):
        if type(m) in [nn.LazyLinear, nn.LazyConv2d]:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.0001)

    def apply_init(self, inputs, init=None):
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)

    def layer_summary(self, X_shape:tuple):
        X = torch.rand(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, "output shape: ", X.shape)

