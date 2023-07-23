import torch.nn as nn


class Cifar10_CNN(nn.Module):
    def __init__(self, channels=3, img_dim=32, outneurons=10, last_hidden_neurons=80,
        weight_init="kaiming_uniform", bias=True, dropout=0.0, batchnorm=True):

        super(Cifar10_CNN, self).__init__()

        self.channels = channels
        self.img_dim = img_dim
        self.in_features = channels * img_dim * img_dim
        self.num_classes = outneurons
        self.last_hidden_neurons = last_hidden_neurons
        self.dropout_p = dropout
        self.batchnorm = batchnorm

        # uniform(-1/sqrt(in_features), 1/sqrt(in_features))
        weights = {
            "normal": nn.init.normal_,
            "xavier": nn.init.xavier_normal_,
            "xavier_uniform": nn.init.xavier_uniform_,
            "kaiming": nn.init.kaiming_normal_,
            "kaiming_uniform": nn.init.kaiming_uniform_,
        }

        self.conv = nn.Sequential(
            nn.Conv2d(self.channels, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(256, 256, 3, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(256, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(128, 128, 3, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p)
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*24*24, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout1d(self.dropout_p),

            nn.Linear(2048, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout1d(self.dropout_p),

            nn.Linear(1024, self.last_hidden_neurons, bias=False)
        )

        self.bn = nn.BatchNorm1d(self.last_hidden_neurons)

        # expand all layers in sequential order
        self.output = nn.Linear(self.last_hidden_neurons, self.num_classes, bias=False)

        # if weight_init:
        #     self.__weight_init(weights[weight_init], bias)

    def forward(self, x):

        x = self._train(x)

        x = self.bn(x)
        x = nn.ReLU()(x)
        x = nn.Dropout1d(self.dropout_p)(x)

        x = self.output(x)

        return x


    def _train(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x

    def output_last_layer(self, x):

        x = self._train(x)
        out = x.clone().detach()

        x = nn.BatchNorm1d(self.last_hidden_neurons)(x)
        x = nn.ReLU()(x)
        x = nn.Dropout1d(self.dropout_p)(x)

        x = self.output(x)

        return out, x


    def _sum_weights(self):

        total_weights = 0
        for _, p in self.named_parameters():
            total_weights += p.sum()

        return total_weights.item()

    def _sum_abs_weights(self):

        total_weights = 0
        for _, p in self.named_parameters():
            total_weights += p.abs().sum()

        return total_weights.item()

    def _l1_regularization(self, alpha=1e-3):

        total_weights = 0
        for _, p in self.named_parameters():
            total_weights += p.abs().sum()

        return alpha * total_weights

    def _l2_regularization(self, lambd=1e-3):

        total_weights = 0
        for _, p in self.named_parameters():
            total_weights += p.pow(2).sum()

        return lambd * total_weights

    def _elastic_regularization(self, lambd=1e-3, alpha=1e-3):
        return self._l2_regularization(lambd) + self._l1_regularization(alpha)

    def __weight_init(self, fn, bias):

        for m in self.modules():

            if (
                isinstance(m, nn.Linear)
                or isinstance(m, nn.Conv2d)
                or isinstance(m, nn.Conv1d)
               ):
                fn(m.weight)

                if bias:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
