import numpy as np
from si.data.dataset import Dataset

class PCA:
    def __init__(self, n_components: int):
        self.n_components = n_components

        self.mean = None
        self.components = None
        self.explained_variance = None

    #Before implementing fit method of PCA: SVD method

    def _centered_data(self, dataset: Dataset) -> np.ndarray:
        self.mean = np.mean(dataset.X, axis = 0) #mean for each column
        
        return dataset.X - self.mean
    
    def _get_principal_components(self, dataset: Dataset) -> np.ndarray:
        centered_data = self._centered_data(dataset)
        
        #calculate SVD
        self.u, self.s, self.v = np.linalg.svd(centered_data, full_matrices=False)

        #components corresponde to the first n_components of V^T
        self.components = self.v[:, :self.n_components] # get the first n_components columns

        return self.components
    
    def _get_explained_variance(self, dataset:Dataset)-> np.ndarray:
        ev = (self.s ** 2)/(len(dataset.X)-1)

        #The explained variance (explained_variance) corresponds to the first n_components of EV
        explained_variance = ev[:self.explained_variance]

    def fit(self, dataset: Dataset) -> 'PCA':
        self.components = self._get_principal_components(dataset)
        self.explained_variance = self._get_explained_variance(dataset)

        return self
    
    def transform(self, dataset:Dataset)-> Dataset:
        if self.components is None:
            raise Exception ("You should 'fit' PCA before calling 'transform")
        centered_data = self._centered_data(dataset)
        
        return np.dot(centered_data, self.components.T)

    def fit_transform(self, dataset:Dataset)-> np.ndarray:
        self.fit(dataset)
        return self.transform(dataset)
    
if __name__ == 'main':
    from si.io.csv_file import read_csv_file

    print("EX1")
    ds = Dataset.from_random(n_examples=10, n_features=10, label=False, seed=0)
    pca = PCA(n_components=4)
    pca.fit(ds)
    print(pca.mean)
    print(pca._get_principal_components)
    print(pca._get_explained_variance)
    x_reduced = pca.transform(ds)
    print(x_reduced)
    
    print("\nEX2 - iris")
    path = "../../../datasets/iris/iris.csv"
    iris = read_csv_file(file=path, sep=",", features=True, label=True)
    pca_iris = PCA(n_components=2)
    iris_reduced = pca_iris.fit_transform(iris)
    print(iris_reduced)

