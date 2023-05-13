import torch.nn as nn


class MNIST_Model(nn.Module):
    def __init__(
        self, img_dim=28, outneurons=10, last_hidden_neurons=30, first_layer_norm=True,
        weight_init='kaiming_uniform', bias=False, dropout=0.0, batchnorm=True
    ):

        super(MNIST_Model, self).__init__()

        self.channels = 1
        self.first_layer_norm = first_layer_norm
        self.batchnorm = batchnorm
        self.img_dim = img_dim
        self.in_features = img_dim * img_dim
        self.num_classes = outneurons
        self.last_hidden_neurons = last_hidden_neurons
        self.dropout_p = dropout

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
        self.scaleInputs = nn.BatchNorm2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, bias=bias),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),

            nn.Conv2d(32, 16, 3, stride=2, bias=bias),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(self.dropout_p),
        )

        self.fc3 = nn.Linear(16 * 12 * 12, 200, bias=bias)
        self.bn3 = nn.BatchNorm1d(200)

        self.fc4 = nn.Linear(200, last_hidden_neurons, bias=bias)
        self.bn4 = nn.BatchNorm1d(last_hidden_neurons)

        self.dropout_l = nn.Dropout(self.dropout_p)

        self.output = nn.Linear(last_hidden_neurons, outneurons, bias=bias)


        if weight_init:
            self.__weight_init(weights[weight_init], bias)

    def forward(self, x):

        x = self._train(x)
        if self.batchnorm: x = self.bn4(x)
        x = self.relu(x)
        x = self.output(x)

        return x

    def _train(self, x):

        if self.first_layer_norm: x = self.scaleInputs(x)

        x = self.conv(x)

        x = self.flatten(x)

        x = self.dropout_l(x)
        x = self.fc3(x)
        if self.batchnorm: x = self.bn3(x)
        x = self.relu(x)

        x = self.dropout_l(x)
        x = self.fc4(x)

        return x

    def output_last_layer(self, x):

        x = self._train(x)
        out = x.clone().detach()
        if self.batchnorm: x = self.bn4(x)
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
