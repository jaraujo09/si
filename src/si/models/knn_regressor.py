from si.data.dataset import Dataset
from si.metrics import rmse
from si.model_selection import split
from si.statistics import euclidean_distance
import numpy as np
from typing import Callable, Union

class KNNRegressor:
    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN Regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use - euclidean distance
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.train_dataset = None

    def fit(self, dataset:Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self
    
    def _get_closest_label(self, sample: np.ndarray) -> float:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        distances = self.distance(sample, self.train_dataset.X)   #distance between each sample
        k_similar = np.argsort(distances)[:self.k]    #indexes of the k most similar examples in crescent order
        k_similar_values = self.train_dataset.y[k_similar]  

        match_class_mean = np.mean(k_similar_values)
        
        return match_class_mean
    
    def predict(self, dataset:Dataset)->np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self._get_closest_label, axis = 1, arr = dataset.X)
        #applying the function to all lines of the dataset (not specific to test or train)
        #checks distances between samples, selects k nearest and does the mean of y values of those

    def score(self, dataset: Dataset) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)
    

