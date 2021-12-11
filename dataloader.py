from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from PIL import Image
import random
from os.path import join
from tqdm import tqdm
import numpy as np
import parameters


def get_class_balanced_labels(targets, labels_per_class, save_path=None):
    num_classes = max(targets) + 1

    indices = list(range(len(targets)))
    random.shuffle(indices)

    label_count = {i: 0 for i in range(num_classes)}
    label_indices, unlabel_indices = [], []
    for idx in indices:
        if label_count[targets[idx].item()] < labels_per_class:
            label_indices.append(idx)
            label_count[targets[idx].item()] += 1
        else:
            unlabel_indices.append(idx)

    if save_path is not None:
        with open(join(save_path, 'label_indices.txt'), 'w+') as f:
            for idx in label_indices:
                f.write(str(idx) + '\n')

    return label_indices, unlabel_indices


def get_repeated_indices(indices, num_iters, batch_size):
    length = num_iters * batch_size
    num_epochs = length // len(indices) + 1
    repeated_indices = []

    for epoch in tqdm(range(num_epochs), desc='Pre-allocating indices'):
        random.shuffle(indices)
        repeated_indices += indices

    return repeated_indices[:length]


class MNIST(dsets.MNIST):
    num_classes = 10

    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
        super(MNIST, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel

        self.label_indices, self.unlabel_indices = get_class_balanced_labels(
            self.targets, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(
            self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(
                self.unlabel_indices, num_iters, batch_size)

    def __len__(self):
        return len(self.repeated_label_indices)

    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], self.targets[label_idx]
        label_img = np.asarray(label_img)
        # print(type(label_img))
        label_img = Image.fromarray(label_img)

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)

        if self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], self.targets[unlabel_idx]
            unlabel_img = np.asarray(unlabel_img)
            unlabel_img = Image.fromarray(unlabel_img)

            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target


class FMNIST(dsets.FashionMNIST):
    num_classes = 10

    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
        super(FMNIST, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel

        self.label_indices, self.unlabel_indices = get_class_balanced_labels(
            self.targets, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(
            self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(
                self.unlabel_indices, num_iters, batch_size)

    def __len__(self):
        return len(self.repeated_label_indices)

    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], self.targets[label_idx]
        label_img = np.asarray(label_img)
        # print(type(label_img))
        label_img = Image.fromarray(label_img)

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)

        if self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], self.targets[unlabel_idx]
            unlabel_img = np.asarray(unlabel_img)
            unlabel_img = Image.fromarray(unlabel_img)

            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target


class STL10(dsets.STL10):
    num_classes = 10

    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
        super(STL10, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel

        self.label_indices, self.unlabel_indices = get_class_balanced_labels(
            self.labels, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(
            self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(
                self.unlabel_indices, num_iters, batch_size)

    def __len__(self):
        return len(self.repeated_label_indices)

    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], int(
            self.labels[label_idx])
        label_img = Image.fromarray(np.transpose(label_img, (1, 2, 0)))

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)

        if self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], int(
                self.labels[unlabel_idx])
            unlabel_img = Image.fromarray(np.transpose(unlabel_img, (1, 2, 0)))

            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target


def dataloader(dset, path, num_iters, num_labels=4000, bs=100, return_unlabel=True, save_path=None):
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

    train_dataset = parameters.train_dset[dset](
        root=path,
        num_labels=num_labels,
        num_iters=num_iters,
        batch_size=bs,
        return_unlabel=return_unlabel,
        transform=parameters.train_transform[dset],
        save_path=save_path,
        **parameters.train_kwargs[dset]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        num_workers=4,
        shuffle=False
    )

    test_dataset = parameters.test_dset[dset](
        root=path,
        transform=parameters.test_transform[dset],
        **parameters.test_kwargs[dset]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=100,
        num_workers=4,
        shuffle=False
    )

    return iter(train_loader), test_loader
