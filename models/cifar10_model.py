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


        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        # scaling data
        self.scaleInputs = nn.BatchNorm2d(channels)
        self.dropout_l = nn.Dropout(self.dropout_p)
        self.pool = nn.MaxPool2d(2, 2)

        self.cn1 = nn.Conv2d(channels, 600, 3, padding=3, bias=bias)
        self.bn1 = nn.BatchNorm2d(600)

        self.cn2 = nn.Conv2d(600, 600, 3, bias=bias)
        self.bn2 = nn.BatchNorm2d(600)

        self.cn3 = nn.Conv2d(600, 300, 3, bias=bias)
        self.bn3 = nn.BatchNorm2d(300)

        self.cn4 = nn.Conv2d(300, 300, 3, padding=3, bias=bias)
        self.bn4 = nn.BatchNorm2d(300)

        self.cn5 = nn.Conv2d(300, 300, 3, bias=bias)
        self.bn5 = nn.BatchNorm2d(300)

        self.cn6 = nn.Conv2d(300, 100, 3, bias=bias)
        self.bn6 = nn.BatchNorm2d(100)

        self.fc7 = nn.Linear(100 * 8 * 8, 100, bias=bias)
        self.bn7 = nn.BatchNorm1d(100)

        self.fc8 = nn.Linear(100, 300, bias=bias)
        self.bn8 = nn.BatchNorm1d(300)

        self.fc9 = nn.Linear(300, last_hidden_neurons, bias=bias)
        self.bn9 = nn.BatchNorm1d(last_hidden_neurons)

        self.output = nn.Linear(last_hidden_neurons, outneurons, bias=bias)

        if weight_init:
            self.__weight_init(weights[weight_init], bias)

    def forward(self, x):

        x = self._train(x)
        if self.batchnorm: x = self.bn9(x)
        x = self.relu(x)
        x = self.output(x)

        return x


    def _train(self, x):

        x = self.scaleInputs(x)

        x = self.relu(self.bn1(self.cn1(x)))
        x = self.relu(self.bn2(self.cn2(x)))
        x = self.pool(self.relu(self.bn3(self.cn3(x))))

        x = self.relu(self.bn4(self.cn4(x)))
        x = self.relu(self.bn5(self.cn5(x)))
        x = self.pool(self.relu(self.bn6(self.cn6(x))))

        x = self.flatten(x)
        x = self.dropout_l(x)
        x = self.relu(self.bn7(self.fc7(x)))
        x = self.dropout_l(x)
        x = self.relu(self.bn8(self.fc8(x)))

        x = self.dropout_l(x)
        x = self.fc9(x)

        return x

    def output_last_layer(self, x):

        x = self._train(x)
        out = x.clone().detach()
        if self.batchnorm: x = self.bn9(x)
        x = self.relu(x)
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
