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
            T.Normalize((0.3381, 0.3101, 0.3194), (0.1625, 0.1625, 0.1721))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.3381, 0.3101, 0.3194), (0.1625, 0.1625, 0.1721))
        ])
    },

    'cifar10': {
        'train': T.Compose([
            T.RandomApply([
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip()
            ], p=0.2),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]),
        'test': T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    }
}