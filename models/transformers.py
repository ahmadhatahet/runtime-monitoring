import torchvision.transforms as T

transformers = {
    'mnist': {
        'train': T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307), (0.3015))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.1307), (0.3015))
        ])
    },

    'fashionmnist': {
        'train': T.Compose([
            T.ToTensor(),
            T.Normalize((0.2850), (0.3200))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.2850), (0.3200))
        ])
    },

    'gtsrb': {
        'train': T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),
            T.Normalize((0.3359, 0.3110, 0.3224), (0.3359, 0.3110, 0.3224))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),
            T.Normalize((0.3359, 0.3110, 0.3224), (0.3359, 0.3110, 0.3224))
        ])
    },

    'cifar10': {
        'train': T.Compose([
            T.ToTensor(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }
}