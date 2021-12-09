import math
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from PIL import Image
import os, random, pickle
from os.path import join, isfile
from tqdm import tqdm
import numpy as np

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

        self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.targets, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)

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


# class iNaturalist(dsets.inaturalist.INaturalist):
#     num_classes = 11
#     def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
#         super(iNaturalist, self).__init__(**kwargs)
#         labels_per_class = num_labels // self.num_classes
#         self.return_unlabel = return_unlabel

#         self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.target, labels_per_class, save_path)
#         self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
#         if self.return_unlabel:
#             self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)

#     def __len__(self):
#         return len(self.repeated_label_indices)

#     def __getitem__(self, idx):
#         label_idx = self.repeated_label_indices[idx]
#         label_img, label_target = self.data[label_idx], int(self.labels[label_idx])
#         label_img = Image.fromarray(np.transpose(label_img, (1, 2, 0)))

#         if self.transform is not None:
#             label_img = self.transform(label_img)
#         if self.target_transform is not None:
#             label_target = self.target_transform(label_target)

#         if self.return_unlabel:
#             unlabel_idx = self.repeated_unlabel_indices[idx]
#             unlabel_img, unlabel_target = self.data[unlabel_idx], int(self.labels[unlabel_idx])
#             unlabel_img = Image.fromarray(np.transpose(unlabel_img, (1, 2, 0)))

#             if self.transform is not None:
#                 unlabel_img = self.transform(unlabel_img)
#             if self.target_transform is not None:
#                 unlabel_target = self.target_transform(unlabel_target)
#             return label_img, label_target, unlabel_img, unlabel_target
#         else:
#             return label_img, label_target


class FMNIST(dsets.FashionMNIST):
    num_classes = 10
    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
        super(FMNIST, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel

        self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.targets, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)

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

        self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.labels, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)

    def __len__(self):
        return len(self.repeated_label_indices)

    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], int(self.labels[label_idx])
        label_img = Image.fromarray(np.transpose(label_img, (1, 2, 0)))

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)

        if self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], int(self.labels[unlabel_idx])
            unlabel_img = Image.fromarray(np.transpose(unlabel_img, (1, 2, 0)))

            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target


class LFW(dsets.LFWPeople):
    num_classes = 5749
    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
        super(LFW, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel

        self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.targets, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)

    def __len__(self):
        return len(self.repeated_label_indices)

    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], self.targets[label_idx]
        label_img = Image.fromarray(np.transpose(label_img, (1, 2, 0)))

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)

        if self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], self.targets[label_idx]
            unlabel_img = Image.fromarray(np.transpose(unlabel_img, (1, 2, 0)))

            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target

# class KITTI(dsets.Kitti):
#     num_classes = 29
#     def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
#         super(KITTI, self).__init__(**kwargs)
#         labels_per_class = num_labels // self.num_classes
#         self.return_unlabel = return_unlabel
#         # print(">>>>>>>>>>>>>>>>>>>" + str(self.targets))
#         self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.labels, labels_per_class, save_path)
#         self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
#         if self.return_unlabel:
#             self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)

#     def __len__(self):
#         return len(self.repeated_label_indices)

#     def __getitem__(self, idx):
#         label_idx = self.repeated_label_indices[idx]
#         label_img, label_target = self.data[label_idx], int(self.labels[label_idx])
        
#         label_img = Image.fromarray(np.transpose(label_img, (1, 2, 0)))

#         if self.transform is not None:
#             label_img = self.transform(label_img)
#         if self.target_transform is not None:
#             label_target = self.target_transform(label_target)

#         if self.return_unlabel:
#             unlabel_idx = self.repeated_unlabel_indices[idx]
#             unlabel_img, unlabel_target = self.data[unlabel_idx], int(self.labels[unlabel_idx])
#             unlabel_img = Image.fromarray(np.transpose(unlabel_img, (1, 2, 0)))

#             if self.transform is not None:
#                 unlabel_img = self.transform(unlabel_img)
#             if self.target_transform is not None:
#                 unlabel_target = self.target_transform(unlabel_target)
#             return label_img, label_target, unlabel_img, unlabel_target
#         else:
#             return label_img, label_target

class KITTI(dsets.Kitti):
    num_classes = 29
    def __init__(self, num_labels, num_iters, batch_size, return_unlabel=True, save_path=None, **kwargs):
        super(KITTI, self).__init__(**kwargs)
        labels_per_class = num_labels // self.num_classes
        self.return_unlabel = return_unlabel
        # print(">>>>>>>>>>>>>>>>>>>" + str(self.targets))
        self.label_indices, self.unlabel_indices = get_class_balanced_labels(self.labels, labels_per_class, save_path)
        self.repeated_label_indices = get_repeated_indices(self.label_indices, num_iters, batch_size)
        if self.return_unlabel:
            self.repeated_unlabel_indices = get_repeated_indices(self.unlabel_indices, num_iters, batch_size)

    def __len__(self):
        return len(self.repeated_label_indices)

    def __getitem__(self, idx):
        label_idx = self.repeated_label_indices[idx]
        label_img, label_target = self.data[label_idx], int(self.labels[label_idx])
        
        label_img = Image.fromarray(np.transpose(label_img, (1, 2, 0)))

        if self.transform is not None:
            label_img = self.transform(label_img)
        if self.target_transform is not None:
            label_target = self.target_transform(label_target)

        if self.return_unlabel:
            unlabel_idx = self.repeated_unlabel_indices[idx]
            unlabel_img, unlabel_target = self.data[unlabel_idx], int(self.labels[unlabel_idx])
            unlabel_img = Image.fromarray(np.transpose(unlabel_img, (1, 2, 0)))

            if self.transform is not None:
                unlabel_img = self.transform(unlabel_img)
            if self.target_transform is not None:
                unlabel_target = self.target_transform(unlabel_target)
            return label_img, label_target, unlabel_img, unlabel_target
        else:
            return label_img, label_target

train_transform = {
        'mnist': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
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
        'inaturalist': transforms.Compose([
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()                                       
        ]),
        'stl10': transforms.Compose([
            #   transforms.RandomResizedCrop(256),
              transforms.RandomHorizontalFlip(),
            #   transforms.CenterCrop(224),
              transforms.ToTensor(),
            #   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'kitti': transforms.Compose([
            #   transforms.RandomResizedCrop(256),
              transforms.RandomHorizontalFlip(),
            #   transforms.CenterCrop(224),
              transforms.ToTensor(),
            #   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        'lfw': transforms.Compose([
            #   transforms.RandomResizedCrop(256),
              transforms.RandomHorizontalFlip(),
            #   transforms.CenterCrop(224),
              transforms.ToTensor(),
            #   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ]),
        }



train_dset = {
        'mnist': MNIST,
        'fmnist': FMNIST,
        'stl10': STL10,
        'kitti': KITTI,
        'lfw': LFW,
        }

test_dset = {
        'mnist': dsets.MNIST,
        'fmnist': dsets.FashionMNIST,
        'stl10': dsets.STL10,
        'kitti': dsets.Kitti,
        'Lfw':dsets.LFWPeople,
        }

train_kwargs = {
        'mnist': {'train': True, 'download': True},
        'fmnist': {'train': True, 'download': True},
        'stl10': {'split': 'train', 'download': True},
        'kitti': {'train': True, 'download': True},
        'lfw': {'split': 'train', 'download': True},
        }

test_kwargs = {
        'mnist': {'train': False, 'download': True},
        'fmnist': {'train': False, 'download': True},
        'stl10': {'split': 'test', 'download': True},
        'kitti': {'train': False, 'download': True},
        'lfw': {'split': 'test', 'download': True},
        }

def dataloader1(dset, path, bs, num_workers, num_labels, num_iters, return_unlabel=True, save_path=None):
    assert dset in ["mnist", "fmnist", "inaturalist", "stl10", "kitti", 'lfw']

    train_dataset = train_dset[dset](
            root = path,
            num_labels = num_labels,
            num_iters = num_iters,
            batch_size = bs,
            return_unlabel = return_unlabel,
            transform = train_transform[dset],
            save_path = save_path,
            **train_kwargs[dset]
    )
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=num_workers, shuffle=False)

    test_transform = transforms.Compose([
            transforms.ToTensor()
            # transforms.Normalize(*meanstd[dset])
    ])
    test_dataset = test_dset[dset](root=path, transform=test_transform, **test_kwargs[dset])
    test_loader = DataLoader(test_dataset, batch_size=100, num_workers=num_workers, shuffle=False)

    return iter(train_loader), test_loader

train_loader, test_loader = dataloader1(
        dset = 'fmnist',
        path = './content/drive/MyDrive/eecs545dataset',
        bs = 1024,
        num_workers = 8,
        num_labels = 2000,
        num_iters = 40000,
        return_unlabel = True,
        save_path = './content/drive/MyDrive/eecs545dataset'
)

a,b,c,d = next(train_loader)
a

next(train_loader)