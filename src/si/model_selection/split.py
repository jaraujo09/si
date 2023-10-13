from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test


def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42)->Tuple[Dataset, Dataset]:
    """
    Splits the dataset into training and testing sets in a stratified manner.

    Parameters
    ----------
    dataset: Dataset
        The dataset to be split.
    test_size: float
        The proportion of the dataset to be used for testing.
    random_state: int
        The seed for random number generation.

    Returns
    -------
    train: Dataset
        The training dataset.
    test: Dataset
        The testing dataset.
    """
    np.random.seed(random_state)

    unique_labels, counts = np.unique(dataset.y, return_counts=True)  #finds the unique elements in an array and the indexes of those
    train_idx = []
    test_idx = []
    for label, count in zip(unique_labels, counts):  #the zip functions allows me to access the class and the number of counts of the class each iteration
        test_samples = int(count * test_size)   
        label_indexes = np.where(dataset.y == label)[0]  #indexes from the current label
        np.random.shuffle(label_indexes)
        test_idx.extend(label_indexes[:test_samples])
        train_idx.extend(label_indexes[test_samples:])

    train = Dataset(dataset.X[train_idx],dataset.y[train_idx],features=dataset.features, label=dataset.label) #changes in X and y, adapted to train and test
    test = Dataset(dataset.X[test_idx],dataset.y[test_idx],features=dataset.features, label=dataset.label)

    return train, test


