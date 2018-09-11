import os.path

from . import DataTransformer
from torch.utils.data import DataLoader
import torchvision.datasets

def create_data_loaders(train_transformation,
                        eval_transformation,
                        datadir, 
                        train_subdir, 
                        eval_subdir,
                        labels,
                        batch_size,
                        labeled_batch_size,
                        workers):
    traindir = os.path.join(datadir, train_subdir)
    evaldir = os.path.join(datadir, eval_subdir)

    dataset = torchvision.datasets.ImageFolder(traindir, train_transformation)

    with open(labels) as f:
        labels = dict(line.split(' ') for line in f.read().splitlines())
    labeled_idxs, unlabeled_idxs = DataTransformer.relabel_dataset(dataset, labels)

    batch_sampler = DataTransformer.TwoStreamBatchSampler(unlabeled_idxs, labeled_idxs, batch_size, labeled_batch_size)
    

    train_loader = DataLoader(dataset,
                              batch_sampler = batch_sampler,
                              num_workers = workers,
                              pin_memory=True)

    eval_loader = DataLoader(
        torchvision.datasets.ImageFolder(evaldir, eval_transformation),
        batch_size = batch_size,
        shuffle = False,
        num_workers = 2 * workers,  # Needs images twice as fast
        pin_memory = True,
        drop_last = False)

    return train_loader, eval_loader