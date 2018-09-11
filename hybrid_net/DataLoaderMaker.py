import os.path
import itertools

import numpy as np

from . import DataTransformer
from torch.utils.data import DataLoader
import torchvision.datasets
from torch.utils.data.sampler import Sampler


# shuffle the iterable object once
def iterate_once(iterable):
    return np.random.permutation(iterable)



# shuffle the iterable object infinite times
def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())



class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    primary indices -- no labels
    secondary indices -- labels 
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


# class TwoStreamBatchSampler(Sampler):
#     """Iterate two sets of indices
#     An 'epoch' is one iteration through the primary indices.
#     During the epoch, the secondary indices are iterated through
#     as many times as needed.

#     primary indices -- labels 
#     secondary indices -- no labels
#     """
#     def __init__(self, primary_indices, secondary_indices, batch_size, primary_batch_size):
#         self.primary_indices = primary_indices
#         self.secondary_indices = secondary_indices
#         self.primary_batch_size = primary_batch_size
#         self.secondary_batch_size = batch_size - primary_batch_size


#         assert len(self.primary_indices) >= self.primary_batch_size > 0
#         assert len(self.secondary_indices) >= self.secondary_batch_size > 0

#     def __iter__(self):
#         primary_iter = iterate_once(self.primary_indices)
#         secondary_iter = iterate_eternally(self.secondary_indices)
#         return (
#             primary_batch + secondary_batch
#             for (primary_batch, secondary_batch)
#             in  zip(grouper(primary_iter, self.primary_batch_size),
#                     grouper(secondary_iter, self.secondary_batch_size))
#         )

#     def __len__(self):
#         return len(self.primary_indices) // self.primary_batch_size



def grouper(iterable, n):
    # Collect data into fixed-length chunks or blocks
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)



def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir, 
                        train_subdir, 
                        eval_subdir,
                        labels,
                        batch_size,
                        labeled_batch_size,
                        workers):

    # obtain train and eval directory paths
    traindir = os.path.join(datadir, train_subdir)
    evaldir = os.path.join(datadir, eval_subdir)

    # make initial dataset from the folders
    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    # get labels of images to be used for SSL, others will have no labels
    with open(labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())
    labeled_idxs, unlabeled_idxs = DataTransformer.relabel_dataset(dataset, labels)


    # sample through the dataset and return batches
    batch_sampler = TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, batch_size, labeled_batch_size)
    # batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, labeled_batch_size)

    # Dataloader for training 
    train_loader = DataLoader(dataset,
                              batch_sampler = batch_sampler,
                              num_workers = workers,
                              pin_memory = True)

    # Dataloader for evaluation 
    eval_loader = DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2 * workers,  # Needs images twice as fast
        pin_memory = True,
        drop_last = False)

    return train_loader, eval_loader