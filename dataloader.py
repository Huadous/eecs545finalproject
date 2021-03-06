from torch.utils.data import DataLoader
import torchvision.datasets as dsets
from PIL import Image
import random
from os.path import join
import numpy as np
import parameters

from tqdm import tqdm

# def get_class_balanced_labels(targets, labels_per_class):
#     indices = list(range(len(targets)))
#     random.shuffle(indices)

#     label_count = {i: 0 for i in range(10)}
#     label_indices, unlabel_indices = [], []
#     for idx in indices:
#         if label_count[targets[idx].item()] < labels_per_class:
#             label_indices.append(idx)
#             label_count[targets[idx].item()] += 1
#         else:
#             unlabel_indices.append(idx)

#     return label_indices, unlabel_indices


class MNIST(dsets.MNIST):
    def __init__(self, num_labels,iteration, bs, **kwargs):
        super(MNIST, self).__init__(**kwargs)
        labels_per_class = num_labels // 10
        class_index = list(range(len(self.targets)))
        random.shuffle(class_index)
        shuffled_class_index = class_index
        labeled_index, unlabeled_index = [], []
        counter = [i for i in range(10)]
        for index in shuffled_class_index:
            if counter[self.targets[index].item()] < labels_per_class:
                counter[self.targets[index].item()] += 1
                labeled_index.append(index)
            else:
                unlabeled_index.append(index)
        total_num = iteration * bs
        num_of_labeld = total_num // len(labeled_index) + 1
        num_of_unlabeld = total_num // len(unlabeled_index) + 1
        self.reformed_labeled_index, self.reformed_unlabeled_index = [], []
        for i in range(num_of_labeld):
            random.shuffle(labeled_index)
            self.reformed_labeled_index += labeled_index
        for i in range(num_of_unlabeld):
            random.shuffle(unlabeled_index)
            self.reformed_unlabeled_index += labeled_index
        self.reformed_labeled_index, self.reformed_unlabeled_index = self.reformed_labeled_index[:total_num], self.reformed_unlabeled_index[:total_num]
        

    def __len__(self):
        return len(self.reformed_labeled_index)

    def __getitem__(self, index):
        labeled_index = self.reformed_labeled_index[index]
        labeled_image, labeled_class = self.data[labeled_index], self.targets[labeled_index]
        labeled_image = self.transform(Image.fromarray(np.asarray(labeled_image)))

        unlabeled_index = self.reformed_unlabeled_index[index]
        unlabeled_image, unlabeled_class = self.data[unlabeled_index], self.targets[unlabeled_index]
        unlabeled_image = self.transform(Image.fromarray(np.asarray(unlabeled_image)))

        return labeled_image, labeled_class, unlabeled_image, unlabeled_class


class FMNIST(dsets.FashionMNIST):
    def __init__(self, num_labels, iteration, bs, **kwargs):
        super(FMNIST, self).__init__(**kwargs)
        labels_per_class = num_labels // 10
        class_index = list(range(len(self.targets)))
        random.shuffle(class_index)
        shuffled_class_index = class_index
        labeled_index, unlabeled_index = [], []
        counter = [i for i in range(10)]
        for index in shuffled_class_index:
            if counter[self.targets[index].item()] < labels_per_class:
                counter[self.targets[index].item()] += 1
                labeled_index.append(index)
            else:
                unlabeled_index.append(index)
        total_num = iteration * bs
        num_of_labeld = total_num // len(labeled_index) + 1
        num_of_unlabeld = total_num // len(unlabeled_index) + 1
        self.reformed_labeled_index, self.reformed_unlabeled_index = [], []
        for i in range(num_of_labeld):
            random.shuffle(labeled_index)
            self.reformed_labeled_index += labeled_index
        for i in range(num_of_unlabeld):
            random.shuffle(unlabeled_index)
            self.reformed_unlabeled_index += labeled_index
        self.reformed_labeled_index, self.reformed_unlabeled_index = self.reformed_labeled_index[:total_num], self.reformed_unlabeled_index[:total_num]

    def __len__(self):
        return len(self.reformed_labeled_index)

    def __getitem__(self, index):
        labeled_index = self.reformed_labeled_index[index]
        labeled_image, labeled_class = self.data[labeled_index], self.targets[labeled_index]
        labeled_image = self.transform(Image.fromarray(np.asarray(labeled_image)))

        unlabeled_index = self.reformed_unlabeled_index[index]
        unlabeled_image, unlabeled_class = self.data[unlabeled_index], self.targets[unlabeled_index]
        unlabeled_image = self.transform(Image.fromarray(np.asarray(unlabeled_image)))

        return labeled_image, labeled_class, unlabeled_image, unlabeled_class


class STL10(dsets.STL10):
    def __init__(self, num_labels, iteration, bs, **kwargs):
        super(STL10, self).__init__(**kwargs)
        labels_per_class = num_labels // 10
        print(self.labels)
        class_index = list(range(len(self.labels)))
        random.shuffle(class_index)
        shuffled_class_index = class_index
        labeled_index, unlabeled_index = [], []
        counter = [i for i in range(10)]
        for index in shuffled_class_index:
            if counter[self.labels[index].item()] < labels_per_class:
                counter[self.labels[index].item()] += 1
                labeled_index.append(index)
            else:
                unlabeled_index.append(index)
        total_num = iteration * bs
        num_of_labeld = total_num // len(labeled_index) + 1
        num_of_unlabeld = total_num // len(unlabeled_index) + 1
        self.reformed_labeled_index, self.reformed_unlabeled_index = [], []
        for i in range(num_of_labeld):
            random.shuffle(labeled_index)
            self.reformed_labeled_index += labeled_index
        for i in range(num_of_unlabeld):
            random.shuffle(unlabeled_index)
            self.reformed_unlabeled_index += labeled_index
        self.reformed_labeled_index, self.reformed_unlabeled_index = self.reformed_labeled_index[:total_num], self.reformed_unlabeled_index[:total_num]

    def __len__(self):
        return len(self.reformed_labeled_index)

    def __getitem__(self, index):
        labeled_index = self.reformed_labeled_index[index]
        labeled_image, labeled_class = self.data[labeled_index], int(self.labels[labeled_index])
        labeled_image = self.transform(Image.fromarray(np.transpose(labeled_image, (1, 2, 0))))

        unlabeled_index = self.reformed_unlabeled_index[index]
        unlabeled_image, unlabeled_class = self.data[unlabeled_index], int(
                self.labels[unlabeled_index])
        unlabeled_image = self.transform(Image.fromarray(np.transpose(unlabeled_image, (1, 2, 0))))

        return labeled_image, labeled_class, unlabeled_image, unlabeled_class


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


def dataloader(dset, path, iteration, num_labels=4000, bs=100):
    train_dataset = train_dset[dset](
        root=path,
        num_labels=num_labels,
        iteration=iteration,
        bs=bs,
        transform=parameters.train_transform[dset],
        **parameters.train_kwargs[dset]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        num_workers=4,
        shuffle=False
    )

    test_dataset = test_dset[dset](
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
