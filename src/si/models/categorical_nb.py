from si.data.dataset import Dataset
import numpy as np

class CategoricalNB:
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize the Categorical Naive Bayes (suitable for categorical/discrete data)

        Parameters
        ----------
        smoothing: float
            Laplace smoothing to avoid zero probabilities
        distance: Callable
            The distance function to use - euclidean distance
        """
        #parameter
        self.smoothing = smoothing

        #atributes
        self.class_prior = None
        self.feature_prob = None

    def fit(self, dataset:Dataset)-> 'CategoricalNB':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: CategoricalNB
            The fitted model
        """
        n_samples = dataset.shape[0]
        n_features = dataset.features
        n_classes = dataset.y

        class_counts = len(n_classes)
        feature_counts = len(n_classes , n_features)
        class_prior = len(n_classes)