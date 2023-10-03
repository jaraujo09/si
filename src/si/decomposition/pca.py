import numpy as np
from si.data.dataset import Dataset

class PCA:
    """
    PCA implementation to reduce the dimensionality of a given dataset. It uses SVD (Singular
    Value Decomposition) to do it.
    """
    def __init__(self, n_components: int = 10):
        """
        PCA implementation to reduce the dimensionality of a given dataset. It uses SVD (Singular
        Value Decomposition) to do it.

        Parameters
        ----------
        n_components: int (default=10)
            The number of principal components to be computed

        Attributes
        ----------
        mean: np.ndarray
            The mean value of each feature of the dataset
        components: np.ndarray
            The first <n_components> principal components
        explained_variance: np.ndarray
            The variances explained by the first <n_components> principal components
        """
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None


    def fit(self, dataset: Dataset) -> 'PCA':
        """
        Fits PCA by computing the mean value of each feature of the dataset, the first 
        <n_components> principal components and the corresponding explained variances.
        Returns self.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        #firstly, center data
        self.mean = np.mean(dataset.X, axis = 0) #mean for each column
        self.centered_data = dataset.X - self.mean

        #calcule of SVD
        u,s,vt = np.linalg.svd(self.centered_data, full_matrices=False)
        
        #components corresponde to the first n_components of V^T
        self.components = vt[:self.n_components]

        #The explained variance (explained_variance) corresponds to the first n_components of EV
        ev = (self.s ** 2)/(len(dataset.X)-1)
        explained_variance = ev[:self.n_components]
        
        return self
    
    def transform(self, dataset:Dataset)-> Dataset:
        """
        Transforms the dataset by reducing X (X_reduced = X @ V, V = self.components.T).
        Returns X reduced.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        if self.components is None:
            raise Exception ("You should 'fit' PCA before calling 'transform")
        centered_data = self._centered_data(dataset)

        v_matrix = self.components.T

        # Get transformed data
        transformed_data = np.dot(self.centered_data, v_matrix)
        
        return Dataset(transformed_data, dataset.y, dataset.features_names, dataset.label_name)

    def fit_transform(self, dataset:Dataset)-> np.ndarray:
        """
        Fits PCA and transforms the dataset by reducing X. Returns X reduced.
        
        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)
    