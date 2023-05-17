import torch
from torch.utils.data import Subset
from torchvision import transforms as tvt, datasets
from utils import Preprocess
from collections import namedtuple

ImageDataset = namedtuple('ImageDataset', ['dataset',
                                           'img_shape',
                                           'preprocess_fn',
                                           'valid_indices'])

def create_dataset(root,
                   dataset,
                   num_bits,
                   pad,
                   valid_size,
                   valid_indices=None,
                   class_index = 0,
                   split='train'):
    assert split in ['train', 'valid', 'test']

    preprocess = Preprocess(num_bits)
    c, h, w = (1, 28 + 2 * pad, 28 + 2 * pad)

    if dataset == 'mnist':
        transforms = []

        if split == 'train':
            transforms += [tvt.RandomHorizontalFlip()]

        transforms += [
            tvt.Pad((pad, pad)),
            tvt.ToTensor(),
            preprocess
        ]

        dataset = datasets.MNIST(
            root=root,
            train=(split in ['train', 'valid']),
            transform=tvt.Compose(transforms),
            download=True
        )

        # Filter out all examples that do not belong to the specified class
        class_indices = torch.where(dataset.targets == class_index)[0]
        dataset = Subset(dataset, class_indices)

    elif dataset == 'fashion-mnist':
        transforms = []

        if split == 'train':
            transforms += [tvt.RandomHorizontalFlip()]

        transforms += [
            tvt.Pad((pad, pad)),
            tvt.ToTensor(),
            preprocess
        ]

        dataset = datasets.FashionMNIST(
            root=root,
            train=(split in ['train', 'valid']),
            transform=tvt.Compose(transforms),
            download=True
        )

        # Filter out all examples that do not belong to the specified class
        class_indices = torch.where(dataset.targets == class_index)[0]
        dataset = Subset(dataset, class_indices)

    else:
        raise RuntimeError('Unknown dataset')

    if split == 'train':
        num_train = len(dataset)
        indices = torch.randperm(num_train).tolist()
        train_indices, valid_indices = indices[valid_size:], indices[:valid_size]
        dataset = Subset(dataset, train_indices)
    elif split == 'valid':
        dataset = Subset(dataset, valid_indices)

    print(f'Using {split} data split of size {len(dataset)}')

    return ImageDataset(dataset=dataset,
                        img_shape=(c, h, w),
                        preprocess_fn=preprocess,
                        valid_indices=valid_indices)
