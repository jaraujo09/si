import numpy as np
from si.data.dataset import Dataset

class PCA:
    """
    PCA implementation to reduce the dimensionality of a given dataset. It uses SVD (Singular
    Value Decomposition) to do it.
    """
    def __init__(self, n_components: int = 10) -> None:
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

    def fit(self, dataset:Dataset) -> "PCA":
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
        centered_data = dataset.X - self.mean

        #calcule of SVD
        U,S,Vt =np.linalg.svd (centered_data, full_matrices=False)

        #components corresponde to the first n_components of V^T
        self.components = Vt[:self.n_components]

        #The explained variance (explained_variance) corresponds to the first n_components of EV
        explained_variance =(S ** 2)/(len(dataset.X)-1)
        self.explained_variance = explained_variance[:self.n_components]
        return self

    def transform (self, dataset: Dataset) -> Dataset:
        """
        Transforms the dataset by reducing X (X_reduced = X @ V, V = self.components.T).
        Returns X reduced.

        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        centered_data = dataset.X - self.mean
        V = np.transpose(self.components)

        principal_components = np.dot(centered_data, V)  #transformed data
        return Dataset(principal_components, dataset.y, dataset.features[:self.n_components], dataset.label)

    def fit_transform(self,dataset: Dataset) -> Dataset:
        """
        Fits PCA and transforms the dataset by reducing X. Returns X reduced.
        
        Parameters
        ----------
        dataset: Dataset
            A Dataset object
        """
        self.fit(dataset)
        return self.transform(dataset)


#testing
if __name__ == '__main__':

    from si.data.dataset import Dataset

    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 1, 0])
    print(X.shape)
    
    label = 'target'
    features = ['feature1', 'feature2', 'feature3']
    dataset = Dataset(X, y, features, label)
    
    #to compare with sklearn
    sklearn_pca = PCA(n_components=2)
    sklearn_pca.fit(dataset)

    sklearn_transformed_data = sklearn_pca.transform(dataset)

    my_pca = PCA(n_components=2)
    my_pca.fit(dataset)
    my_transformed_dataset = my_pca.transform(dataset)

    print("scikit-learn Transformed Data:")
    print(sklearn_transformed_data.X)
    print("My PCA Transformed Data:")
    print(my_transformed_dataset.X)

