import torchvision.transforms as transforms

train_transform = {
    'mnist': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.Resize(84),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize(*meanstd['cifar10'])
    ]),
    'fmnist': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
        # transforms.Normalize(*meanstd['cifar10'])
    ]),
    'stl10': transforms.Compose([
        #   transforms.RandomResizedCrop(256),
        transforms.RandomHorizontalFlip(),
        #   transforms.CenterCrop(224),
        transforms.ToTensor(),
        #   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ]),
}

test_transform = {
    'mnist': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ]),
    'fmnist': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor()
    ]),
    'stl10': transforms.Compose([
        transforms.ToTensor()
    ])
}

train_dset = {
    'mnist': MNIST,
    'fmnist': FMNIST,
    'stl10': STL10,
}

test_dset = {
    'mnist': dsets.MNIST,
    'fmnist': dsets.FashionMNIST,
    'stl10': dsets.STL10,
}

train_kwargs = {
    'mnist': {'train': True, 'download': True},
    'fmnist': {'train': True, 'download': True},
    'stl10': {'split': 'train', 'download': True},
}

test_kwargs = {
    'mnist': {'train': False, 'download': True},
    'fmnist': {'train': False, 'download': True},
    'stl10': {'split': 'test', 'download': True},
}