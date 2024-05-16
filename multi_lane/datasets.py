# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for MULTI-LANE
# Thomas De Min thomas.demin@unitn.it
# ------------------------------------------

import random

import torch
from torch.utils.data.dataset import Subset, ConcatDataset
from torchvision import datasets, transforms

from timm.data import create_transform

from multi_lane.continual_datasets.continual_datasets import *

import multi_lane.utils as utils

class Lambda(transforms.Lambda):
    def __init__(self, lambd, nb_classes):
        super().__init__(lambd)
        self.nb_classes = nb_classes
    
    def __call__(self, img):
        return self.lambd(img, self.nb_classes)

def target_transform(x, nb_classes):
    return x + nb_classes

def build_continual_dataloader(args):
    """
    Builds a list of dictionaries with the same length as the number of tasks.
    Each dictionary has 2 keys, train and val.
    Each value is a dataloader.
    """
    dataloader = list()
    class_mask = list() if args.task_inc or args.train_mask else None

    transform_train = build_transform(is_train=True, args=args)
    transform_val = build_transform(is_train=False, args=args)

    if args.dataset.startswith('Split-'):
        dataset_train, dataset_val = get_dataset(args.dataset.replace('Split-',''), transform_train, transform_val, args)

        args.num_classes = len(dataset_val.classes)
        splited_dataset, class_mask = split_single_dataset(dataset_train, dataset_val, args)
    else:
        if args.dataset == '5-datasets':
            dataset_list = ['SVHN', 'MNIST', 'CIFAR10', 'NotMNIST', 'FashionMNIST']
        else:
            dataset_list = args.dataset.split(',')
        
        if args.shuffle == 'yes':
            random.shuffle(dataset_list)
        print(dataset_list)
    
        args.num_classes = 0

    for i in range(args.num_tasks):
        if args.dataset.startswith('Split-'):
            dataset_train, dataset_val = splited_dataset[i]

        else:
            dataset_train, dataset_val = get_dataset(dataset_list[i], transform_train, transform_val, args)

            transform_target = Lambda(target_transform, args.num_classes)

            if class_mask is not None:
                class_mask.append([i + args.num_classes for i in range(len(dataset_val.classes))])
                args.num_classes += len(dataset_val.classes)

            if not args.task_inc:
                dataset_train.target_transform = transform_target
                dataset_val.target_transform = transform_target

        if args.distributed and utils.get_world_size() > 1:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()

            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=args.drop_last,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=24,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
        )

        dataloader.append({
            'train': data_loader_train, 'val': data_loader_val,
        })

    return dataloader, class_mask


def get_dataset(dataset, transform_train, transform_val, args,):
    if dataset == 'CIFAR100':
        dataset_train = datasets.CIFAR100(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR100(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'CIFAR10':
        dataset_train = datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'MNIST':
        dataset_train = MNIST_RGB(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = MNIST_RGB(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'FashionMNIST':
        dataset_train = FashionMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = FashionMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'SVHN':
        dataset_train = SVHN(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = SVHN(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'NotMNIST':
        dataset_train = NotMNIST(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = NotMNIST(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Flower102':
        dataset_train = Flowers102(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = Flowers102(args.data_path, split='test', download=True, transform=transform_val)
    
    elif dataset == 'Cars196':
        dataset_train = StanfordCars(args.data_path, split='train', download=True, transform=transform_train)
        dataset_val = StanfordCars(args.data_path, split='test', download=True, transform=transform_val)
        
    elif dataset == 'CUB200':
        dataset_train = CUB200(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = CUB200(args.data_path, train=False, download=True, transform=transform_val).data
    
    elif dataset == 'Scene67':
        dataset_train = Scene67(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Scene67(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'TinyImagenet':
        dataset_train = TinyImagenet(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = TinyImagenet(args.data_path, train=False, download=True, transform=transform_val).data
        
    elif dataset == 'Imagenet-R':
        dataset_train = Imagenet_R(args.data_path, train=True, download=True, transform=transform_train).data
        dataset_val = Imagenet_R(args.data_path, train=False, download=True, transform=transform_val).data

    elif dataset == 'COCO':
        dataset_train = COCO(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = COCO(args.data_path, train=False, download=True, transform=transform_val)

    elif dataset == 'VOC':
        dataset_train = VOC(args.data_path, train=True, download=True, transform=transform_train)
        dataset_val = VOC(args.data_path, train=False, download=True, transform=transform_val)
    
    elif dataset == 'Places365':
        root = os.path.join(args.data_path, 'places365_standard')
        file_path = lambda x: os.path.join(root, x + '_filt.txt')
        dataset_train = FileDataset(root, file_path("train"), transfom=transform_train)
        dataset_val = FileDataset(root, file_path("val"), transfom=transform_val)

    elif dataset == 'DTD':
        dataset_train = datasets.DTD(args.data_path, split='train', download=True, transform=transform_train)
        dataset_train2 = datasets.DTD(args.data_path, split='val', download=True, transform=transform_train)
        dataset_train = ConcatDataset((dataset_train, dataset_train2))
        dataset_val = datasets.DTD(args.data_path, split='test', download=True, transform=transform_val)

    else:
        raise ValueError('Dataset {} not found.'.format(dataset))
    
    return dataset_train, dataset_val

def split_single_dataset(dataset_train, dataset_val, args):
    """
    Split the dataset in args.num_tasks different tasks.
    Returns a list of tuples where each tuple is a pair train_ds, val_ds for a given task.
    """
    nb_classes = len(dataset_val.classes)
    # Calculate the number of classes per task
    if args.num_tasks == 1:
        classes_per_task = nb_classes
    else:
        # (tot classes - classes in first task) / (num_tasks - 1)
        assert (nb_classes - args.base_classes) % (args.num_tasks - 1) == 0, "Invalid number of tasks"
        classes_per_task = (nb_classes - args.base_classes) // (args.num_tasks - 1)

    labels = [i for i in range(nb_classes)]
    
    split_datasets = list()
    mask = list()

    if args.shuffle == 'yes':
        print('Shuffling classes...')
        random.shuffle(labels)

    for i in range(args.num_tasks):
        train_split_indices = []
        test_split_indices = []
        
        if i == 0:
            scope = labels[:args.base_classes]
            labels = labels[args.base_classes:]
        else:
            scope = labels[:classes_per_task]
            labels = labels[classes_per_task:]

        mask.append(scope)

        for k in range(len(dataset_train.targets)):
            if type(dataset_train.targets[k]) == list:
                if set(dataset_train.targets[k]).intersection(set(scope)) != set():
                    train_split_indices.append(k)
            else:
                if int(dataset_train.targets[k]) in scope:
                    train_split_indices.append(k)

        for k in range(len(dataset_val.targets)):
            if type(dataset_val.targets[k]) == list:
                if set(dataset_val.targets[k]).intersection(set(scope)) != set():
                    test_split_indices.append(k)
            else:
                if int(dataset_val.targets[k]) in scope:
                    test_split_indices.append(k)
        
        subset_train, subset_val = Subset(dataset_train, train_split_indices), Subset(dataset_val, test_split_indices)

        split_datasets.append([subset_train, subset_val])

    return split_datasets, mask

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (args.min_scale, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = [
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
        ]

        transform.append(transforms.ToTensor())
        
        if args.normalize_input:
            transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                  std=[0.229, 0.224, 0.225]))

        return transforms.Compose(transform)

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    if args.normalize_input:
        t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                       std=[0.229, 0.224, 0.225]))
    
    return transforms.Compose(t)