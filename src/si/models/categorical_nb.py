from si.data.dataset import Dataset
import numpy as np

class CategoricalNB:
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize the Categorical Naive Bayes

        Parameters
        ----------
        smoothing: float
            Laplace smoothing to avoid zero probabilities
        distance: Callable
            The distance function to use - euclidean distance
        """