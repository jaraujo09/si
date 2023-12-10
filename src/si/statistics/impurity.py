import numpy as np


def entropy_impurity(y: np.ndarray) -> float:
    """
    Calculates the impurity of a dataset using entropy.

    Parameters
    ----------
    y: np.ndarray
        The labels of the dataset.

    Returns
    -------
    float
        The impurity of the dataset.
    """
    classes, counts = np.unique(y, return_counts=True)
    impurity = 0
    for i in range(len(classes)):
        impurity -= (counts[i] / len(y)) * np.log2(counts[i] / len(y))
    return impurity


def gini_impurity(y: np.ndarray) -> float:
    """
    Calculates the impurity of a dataset using the Gini index.

    Parameters
    ----------
    y: np.ndarray
        The labels of the dataset.

    Returns
    -------
    float
        The impurity of the dataset.
    """
    classes, counts = np.unique(y, return_counts=True)
    impurity = 1
    for i in range(len(classes)):
        impurity -= (counts[i] / len(y)) ** 2
    return impurity


if __name__ == "__main__":
    # Test data
    labels_entropy = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    labels_gini = np.array([0, 1, 0, 1, 1, 0, 0, 1, 1])

    # Test entropy_impurity
    entropy_result = entropy_impurity(labels_entropy)
    print(f"Entropy Impurity for labels_entropy: {entropy_result}")

    # Test gini_impurity
    gini_result = gini_impurity(labels_gini)
    print(f"Gini Impurity for labels_gini: {gini_result}")
