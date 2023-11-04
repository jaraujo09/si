from typing import Literal, Tuple, Union
import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models.decision_tree_classifier import DecisionTreeClassifier

class RandomForestClassifier:
    """
    Ensemble machine learning technique that combines multiple decision trees to improve prediction accuracy and reduce overfitting
    """

    def __init__(self, n_estimators:int = 1000, 
                 max_features:int = None,
                 min_sample_split:int = 2,
                 max_depth:int = 15,
                 mode: Literal['gini','entropy'] = 'gini',
                 seed:int = 42):
        """
        Uses a collection of decision trees that trains on random subsets of the data using a random subsets of the features.

        Parameters
        ----------
        n_estimators: int
            number of decision trees to use.
        max_features: int
            maximum number of features to use per tree. 
        min_sample_split: int
            minimum samples allowed in a split.
        max_depth: int
            maximumdepth of the trees.
        mode: Literal['gini', 'entropy']
            the mode to use for calculating the information gain.
        seed: int
            random seed to use to assure reproducibility
        """
        #attributes
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed

        #parameters
        self.trees = []
        self.training = {}

    def fit(self, dataset:Dataset)->'RandomForestClassifier':
        """
        Fits the random forest classifier to a dataset.
        train the decision trees of the random forest
        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        n_samples, n_features = dataset.shape()
        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features)) 
        #creating a bootstrap where the chosen samples are repetead but not the features
        for x in range(self.n_estimators):  #number of trees to use
            bootstrap_samples = np.random.choice(n_samples, n_samples, replace = True) #randomly selects n_samples indexs with replacements, which means that the same sample can be selected multiple times
            bootstrap_features = np.random.choice(n_features, self.max_features, replace=False) #randomly selects n_features up to max_features indices, ensuring that the same feature is not selected more than once
            # for example, samples [a,b,c,d] and features [f1,f2,f3,f4] we can get something like [a,f1], [a,f4], [b,f3]
            random_dataset = Dataset(dataset.X[bootstrap_samples][:,bootstrap_features], dataset.y[bootstrap_samples])
        
            tree = DecisionTreeClassifier(min_sample_split=self.min_sample_split, max_depth=self.max_depth, mode = self.mode)

            tree.fit(random_dataset)

            self.trees.append((bootstrap_features, tree))  #tuple containing the features used and the trained tree

        return self
    
    def predict(self, dataset:Dataset)-> np.ndarray:
        """
        Predicts the class labels for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset for which to make predictions.

        Returns
        -------
        np.ndarray
            An array of predicted class labels.
        """
        n_samples = dataset.shape()[0]
        predictions = np.zeros((self.n_estimators, n_samples), dtype=object) #rows are trees and columns the predictions for each sample

        #for each tree
        row = 0
        for feature_idx, tree in enumerate(self.trees):
            feature_idx = self.trees[feature_idx][0]
            tree_nr = self.trees[feature_idx][1]
            data_samples = Dataset(dataset.X[:,feature_idx], dataset.y) #not all dataset, only features in the current tree

            tree_preds = tree_nr.predict(data_samples)  #predict sampled_data only
            predictions[row,:] = tree_preds  #preds of the tree in the current row
            row +=1

        #now get the most commom predicted class
        def majority_vote(sample_predictions):
            unique_classes, counts = np.unique(sample_predictions,return_counts=True)
            most_common = unique_classes[np.argmax(counts)]
            return most_common
        
        majority_vote_prediction = np.apply_along_axis(majority_vote, axis=0, arr=predictions)

        return majority_vote_prediction
    

    def score(self, dataset:Dataset)->float:
        """
        Calculates the accuracy of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to calculate the accuracy on.

        Returns
        -------
        float
            The accuracy of the model on the dataset.
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)

if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split
    filename = r"C:\Users\Fofinha\Desktop\UNI\MESTRADO\2o ANO\Sistemas Inteligentes\si\datasets\iris\iris.csv"

    data = read_csv(filename, sep=",",features=True,label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=10000,max_features=4,min_sample_split=2, max_depth=5, mode='gini',seed=42)
    model.fit(train)
    print(model.score(test))
        