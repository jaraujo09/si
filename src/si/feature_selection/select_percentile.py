import numpy as np

from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification

class SelectPercentile:
    """
    Selects a certain percentage of the features taking into account the F-score value.
    First we see the F-score of each feature and sorted that.
    Then we choose a percentil that representes x % of this f-values sorted
    so we keep the features that indices have the F-value <= to the percentile
    
    Parameters
    -----------
    score_func: taking the dataset and return a pair os array (F and p value)- allow analize the variance 
    percentile: number that represents a percentage of the data 

    estimated parameters(given by the score_func)
    ---------------
    F: 
        the F value for each feature estimated by the score_func
    p:
        the p value for each feature estimated by the score_func
    """
    
    def __init__(self, score_func = f_classification, percentile : float = 50)-> None:
        """
        Select features with the highest F value up to the specified percentile.

        Parameters
        ----------
        score_func: callable, default = f_classification
            Variance analysis function. Function taking dataset and returning a pair of arrays (scores, p_values)

        percentile: float, default = 50
            Percentile for selecting features
        """
        self.score_func = score_func
        self.percentile = percentile
        self.F = None
        self.p = None

    def fit(self, dataset : Dataset)-> 'SelectPercentile':
        """
        It fits SelectPercentile to compute the F scores and p-values.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        self: object
            Returns self.
        """
        self.F, self.p = self.score_func(dataset) #estimatesthe F and p values for each feature using the scoring_func
        return self
    
    def transform(self, dataset: Dataset)-> Dataset:
        """
        It transforms the dataset by selecting the k highest scoring features.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with features with the highest F value up to the specified percentile, with their names.
        """
        nfeat = len(dataset.features)   #calculates the nr of features
        mask = int(nfeat*self.percentile/100)   #nr of features based on percentile (0.2 by default)
        idx = np.argsort(self.F)[-mask:]   #sorts the F-values from the dataset and their corresponding feature names
        best_feat = dataset.X[:, idx]
        best_feat_name = [dataset.features[i] for i in idx]

        return Dataset(best_feat, dataset.y, best_feat_name, dataset.label)
    

    def fit_transform(self, dataset: Dataset) -> Dataset:
        """
        It fits SelectPercentile and transforms the dataset by selecting the features with the highest F value up to the specified percentile.

        Parameters
        ----------
        dataset: Dataset
            A labeled dataset

        Returns
        -------
        dataset: Dataset
            A labeled dataset with the features with the highest F value up to the specified percentile
        """
        self.fit(dataset)
        return self.transform(dataset)
    
#Testing
if __name__ == '__main__':
    from si.data.dataset import Dataset

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                    [0, 1, 4, 3],
                                    [0, 1, 1, 3]]),
                        y=np.array([0, 1, 0]),
                        features=["f1", "f2", "f3", "f4"],
                        label="y")

    selector = SelectPercentile()
    selector = selector.fit(dataset)
    dataset = selector.transform(dataset)
    print(dataset.features)


    percentiles = [50,25,75]
    for percentile in percentiles:
        selector = SelectPercentile(percentile=percentile)
        selector = selector.fit(dataset)
        dataset_filtered = selector.transform(dataset)
        print(f"Features for percentile {percentile}: {dataset_filtered.features}")
        print(dataset_filtered.X)