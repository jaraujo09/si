from typing import Callable, Dict, List

import numpy as np

from si.data.dataset import Dataset


def k_fold_cross_validation(model, dataset: Dataset, scoring: callable = None, cv: int = 3,
                            seed: int = None) -> List[float]:
    """
    Perform k-fold cross-validation on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    cv: int
        The number of cross-validation folds.
    seed: int
        The seed to use for the random number generator.

    Returns
    -------
    scores: List[float]
        The scores of the model on each fold.
    """
    num_samples = dataset.X.shape[0]
    fold_size = num_samples // cv
    scores = []

    # Create an array of indices to shuffle the data
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for fold in range(cv):
        # Determine the indices for the current fold
        start = fold * fold_size
        end = (fold + 1) * fold_size

        # Split the data into training and testing sets
        test_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

        dataset_train = Dataset(dataset.X[train_indices], dataset.y[train_indices])
        dataset_test = Dataset(dataset.X[test_indices], dataset.y[test_indices])

        # Fit the model on the training set and score it on the test set
        model.fit(dataset_train)
        fold_score = scoring(dataset_test.y, model.predict(dataset_test)) if scoring is not None else model.score(
            dataset_test)
        scores.append(fold_score)

    return scores


def leave_one_out_cross_validation(model, dataset: Dataset, scoring: callable = None, seed: int = None) -> List[float]:
    """
    Perform Leave-One-Out Cross-Validation (LOOCV) on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    scoring: Callable
        The scoring function to use. If None, the model's score method will be used.
    seed: int
        The seed to use for the random number generator.

    Returns
    -------
    scores: List[float]
        The scores of the model for each data point.
    """
    num_samples = dataset.X.shape[0]
    scores = []

    # Create an array of indices to shuffle the data
    if seed is not None:
        np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    for i in range(num_samples):
        # Use data point i as the test set, and the rest as the training set
        train_indices = np.delete(indices, i)
        test_indices = [i]

        dataset_train = Dataset(dataset.X[train_indices], dataset.y[train_indices])
        dataset_test = Dataset(dataset.X[test_indices], dataset.y[test_indices])

        # Fit the model on the training set and score it on the test set
        model.fit(dataset_train)
        fold_score = scoring(dataset_test.y, model.predict(dataset_test)) if scoring is not None else model.score(
            dataset_test)
        scores.append(fold_score)

    return scores


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.models.knn_classifier import KNNClassifier

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the KNN
    knn = KNNClassifier(k=3)

    # cross validate the model
    scores_ = k_fold_cross_validation(knn, dataset_, cv=5, seed=1)

    # print the scores
    print(scores_)
    # print mean score and standard deviation
    print(f'Mean score: {np.mean(scores_)} +/- {np.std(scores_)}')

    # LOOCV
    scores_ = leave_one_out_cross_validation(knn, dataset_, seed=1)

    # print the scores
    #print(scores_)
    # print mean score and standard deviation
    print(f'Mean score: {np.mean(scores_)} +/- {np.std(scores_)}')

    print('\n Testing with breast-bin dataset')
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.logistic_regression import LogisticRegression

    filename_breast = r"C:\Users\Fofinha\Desktop\UNI\MESTRADO\2o ANO\Sistemas Inteligentes\si\datasets\breast_bin\breast-bin.csv"
    breast=read_csv(filename_breast, sep=",",features=True,label=True)
    model = LogisticRegression()
    train_data, test_data = stratified_train_test_split(breast, test_size=0.20, random_state=42)
    scores_ = k_fold_cross_validation(model, breast, cv=5, seed=1)
    print(scores_)
    print(f'Mean score: {np.mean(scores_)} +/- {np.std(scores_)}')