import numpy as np
from si.data.dataset import Dataset
from si.statistics.euclidean_distance import euclidean_distance
from si.metrics.accuracy import accuracy
from si.metrics.rmse import rmse
from si.model_selection.split import stratified_train_test_split
from typing import Callable,Union
from si.model_selection.split import train_test_split

class KNNRegressor:
    """
    The k-Nearest Neighbors regressor is a machine learning model that predicts the value of a new sample based on 
    a similarity measure (e.g., distance functions). This algorithm estimates the value of a new sample by
    considering the values(mean) of the k-nearest samples in the training data.

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """
    def __init__(self,k:int=1,distance:Callable=euclidean_distance):
        """
        This algorithm predicts the class for a sample using the k most similar examples.But is suitable for regression problems.
        So estimates the average value of the k most similar examples instead of the most common class.
        Args:
            k :int 
                number of examples to consider
            distance: Callable 
                euclidean distance function. .
        """
        #parameters
        self.k=k # numero de k-mais proximos exemplos a considera
        
        #atributes
        self.distance=distance #distance function between each sample to the samples of the train dataset
        self._train_dataset=None # train dataset

    def fit(self, dataset:Dataset) -> "KNNRegressor":
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
        self._train_dataset=dataset # o input é o dataset de treino logo apenas fiz este passo-guardar o dataset treino
        return self
    
    def _get_closest_value_label(self,sample:np.ndarray)->int:
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
        distances=self.distance(sample,self._train_dataset.X) #distance between each sample to the samples of the train dataset
        k_nearest_neighbors=np.argsort(distances)[:self.k] #indexes of the k most similar examples in crescent order, where the first K will be the closest

        k_nearest_neighbors_values_labels=self._train_dataset.y[k_nearest_neighbors] #  y values of the shortest distance
        return np.mean(k_nearest_neighbors_values_labels) #mesma logica mas agora aplico as medias

    
    def predict(self,dataset:Dataset) -> np.ndarray:
        """
        It predicts the mean label values of the given dataset
        Go to every sample and calculate the distance of each sample(line dataset) to the rest of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model, with the supose values
        """
        return np.apply_along_axis(self._get_closest_value_label,axis=1,arr=dataset.X) #
       
    def score(self,dataset:Dataset) -> float:
        """
        It returns the rmse of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        rmse: float
            The rmse(error) of the model
        """       
        predictions=self.predict(dataset)
        return rmse(dataset.y,predictions) #compares the predicted with real values


if __name__ == '__main__':
    num_samples = 600
    num_features = 100

    X = np.random.rand(num_samples, num_features)
    y = np.random.rand(num_samples)  # Valores apropriados para regressão

    dataset_ = Dataset(X=X, y=y)

    #features and class name 
    dataset_.features = ["feature_" + str(i) for i in range(num_features)]
    dataset_.label = "target"

    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # regressor KNN
    knn_regressor = KNNRegressor(k=5)  

    # fit the model to the train dataset
    knn_regressor.fit(dataset_train)

    # evaluate the model on the test dataset
    score = knn_regressor.score(dataset_test)
    print(f'The rmse of the model is: {score}')