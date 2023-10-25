from typing import Tuple, Sequence, Union

import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray = None, features: Sequence[str] = None, label: str = None) -> None:
        """
        Dataset represents a tabular dataset for single output classification.

        Parameters
        ----------
        X: numpy.ndarray (n_samples, n_features)
            The feature matrix (values that characterize the features)
        y: numpy.ndarray (n_samples, 1)
            The label vector
        features: list of str (n_features)
            The feature names (vector)
        label: str (1)
            The label name (name of the dependent variable, only one)
        """
        if X is None:
            raise ValueError("X cannot be None")  #must exist a matrix X
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if features is not None and len(X[0]) != len(features):
            raise ValueError("Number of features must match the number of columns in X")
        if features is None:
            features = [f"feat_{str(i)}" for i in range(X.shape[1])]
        if y is not None and label is None:
            label = "y"
        self.X = X
        self.y = y
        self.features = features
        self.label = label

    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the dataset, that is a tuple of (samples, features)
        Returns
        -------
        tuple (n_samples, n_features)
        """
        return self.X.shape

    def has_label(self) -> bool:
        """
        Returns True if the dataset has a label (y)
        Returns
        -------
        bool
        """
        return self.y is not None

    def get_classes(self) -> np.ndarray:
        """
        Returns the unique classes in the dataset (possible values of y)
        Returns
        -------
        numpy.ndarray (n_classes)
        """
        if self.has_label():
            return np.unique(self.y)
        else:
            raise ValueError("Dataset does not have a label")

    def get_mean(self) -> np.ndarray:
        """
        Returns the mean of each feature (column)
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmean(self.X, axis=0)

    def get_variance(self) -> np.ndarray:
        """
        Returns the variance of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanvar(self.X, axis=0)

    def get_median(self) -> np.ndarray:
        """
        Returns the median of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmedian(self.X, axis=0)

    def get_min(self) -> np.ndarray:
        """
        Returns the minimum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmin(self.X, axis=0)

    def get_max(self) -> np.ndarray:
        """
        Returns the maximum of each feature
        Returns
        -------
        numpy.ndarray (n_features)
        """
        return np.nanmax(self.X, axis=0)

    def summary(self) -> pd.DataFrame:
        """
        Returns a summary of the dataset, with all descriptive methods
        Returns
        -------
        pandas.DataFrame (n_features, 5)
        """
        data = {
            "mean": self.get_mean(),
            "median": self.get_median(),
            "min": self.get_min(),
            "max": self.get_max(),
            "var": self.get_variance()
        }
        return pd.DataFrame.from_dict(data, orient="index", columns=self.features)


    def dropna(self) -> np.ndarray:
        """
        Method that removes the line of the samples containing at least one null value (NaN), updating X and y.
        """
        na_values = np.isnan(self.X).any(axis = 1)
        self.X = self.X[~na_values]  #negation operator (which means if is not an na_value)
        self.y = self.y[~na_values]
        index = np.where(na_values)

        return self, index

    def fillna(self, choice:str=None):
        """
        Replaces all null values with another value or the mean or median of the feature/variable. 
        
        Parameters:
        -----------
        choice : str
                choose the method to fill the NA (value, mean or median)
        """
        if choice is None:
            raise ValueError("please, put the value or mean or median")
        columns_true=np.isnan(self.X).any(axis=0) 
        nan_columns_indices = np.where(columns_true)[0] # get index where bool is True
        

        for col_index in nan_columns_indices:
            col = self.X[:, col_index]#cols with nan vals
            
            if choice == "value": #opto por escolher entro o maximo e o minimo
                min_value = np.nanmin(col)
                max_value = np.nanmax(col)
                final = np.random.uniform(min_value, max_value) #algo aleatorio entre o minimo e o maximo
            elif choice == "median":
                final = np.nanmedian(col)
            elif choice == "mean":
                final = np.nanmean(col)
            
            col[np.isnan(col)] = final # vou buscar os valores nan como true(dentro da coluna já identificada como ter esses valores) e depois , é basicamente col[onde é true?] e substituir pelo que quero

        return self
    
    def remove_from_index(self, index:int):
        """
        Removes a sample by its index,  updating the y vector by removing the entry associated with the sample to be removed
        
        Parameters:
        -----------
        index : int 
                integer corresponding to the sample to remove
        """
        if not isinstance(index, int):
            raise ValueError("Please provide a valid integer index.")
        
        if index <0 and index> self.X.shape[0]:
            raise ValueError("Write a valid index")
        
        self.X = np.delete(self.X, index, axis=0) 
        if self.y is not None: #if there is a y i want to delete it 
            self.y = np.delete(self.y, index)
        
        return self


    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, label: str = None):
        """
        Creates a Dataset object from a pandas DataFrame

        Parameters
        ----------
        df: pandas.DataFrame
            The DataFrame
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        if label:
            X = df.drop(label, axis=1).to_numpy()
            y = df[label].to_numpy()
        else:
            X = df.to_numpy()
            y = None

        features = df.columns.tolist()
        return cls(X, y, features=features, label=label)

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the dataset to a pandas DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        if self.y is None:
            return pd.DataFrame(self.X, columns=self.features)
        else:
            df = pd.DataFrame(self.X, columns=self.features)
            df[self.label] = self.y
            return df

    @classmethod
    def from_random(cls,
                    n_samples: int,
                    n_features: int,
                    n_classes: int = 2,
                    features: Sequence[str] = None,
                    label: str = None):
        """
        Creates a Dataset object from random data

        Parameters
        ----------
        n_samples: int
            The number of samples
        n_features: int
            The number of features
        n_classes: int
            The number of classes
        features: list of str
            The feature names
        label: str
            The label name

        Returns
        -------
        Dataset
        """
        X = np.random.rand(n_samples, n_features)
        y = np.random.randint(0, n_classes, n_samples)
        return cls(X, y, features=features, label=label)

#Testing
if __name__ == '__main__':
    X = np.array([[1, 2, 3], [4, 5, 6]])
    y = np.array([1, 2])
    features = np.array(['a', 'b', 'c'])
    label = 'y'
    dataset = Dataset(X, y, features, label)
    print('Shape: ', dataset.shape())
    print('Has label: ', dataset.has_label())
    print('Classes: ', dataset.get_classes())
    print('Mean: ', dataset.get_mean())
    print('Variance: ', dataset.get_variance())
    print('Median: ', dataset.get_median())
    print('Minimun: ', dataset.get_min())
    print('Maximun: ', dataset.get_max())
    print('Summary: \n', dataset.summary())
    

    # Dados de exemplo
    X = np.array([[1, 2, 3],
                [4, 5, np.nan],
                [7, np.nan, 9]])
    y = np.array([0, 1, 0])
    features = ['feature_1', 'feature_2', 'feature_3']
    label = 'target'
    dataset = Dataset(X, y, features, label)

    choice = 'value'
    dataset_choice = dataset.fillna(choice)
    print('Fill NA: \n', dataset_choice.X)
    print('Dataset: \n', dataset.X,dataset.y)
