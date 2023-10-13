from si.data.dataset import Dataset
import numpy as np
from si.models.knn_classifier import * 

def rmse(y_true:int, Y_pred:int)->float:
    """
    It returns the mean squared error of the model on the given dataset

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset
    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    mse: float
        The mean squared error of the model
    """

    return np.sqrt(np.sum((y_true-Y_pred)**2) / len(y_true))