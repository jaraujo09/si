from typing import Literal, Tuple, Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.impurity import gini_impurity, entropy_impurity


class Node:
    """
    Class representing a node in a decision tree.
    """

    def __init__(self, feature_idx: int = None, threshold: float = None, left: 'Node' = None, right: 'Node' = None,
                 info_gain: float = None, value: Union[float, str] = None) -> None:
        """
        Creates a Node object.

        Parameters
        ----------
        feature_idx: int
            The index of the feature to split on.
        threshold: float
            The threshold value to split on. (decides if i go to left or right)
        left: Node
            The left subtree.
        right: Node
            The right subtree.
        info_gain: float
            The information gain. (which split is the best)
        value: Union[float, str]
            The value of the leaf node. (only leaf, end of the tree)
        """
        # for decision nodes (middle nodes)
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        # for leaf nodes (final leafs)
        self.value = value


class DecisionTreeClassifier:
    """
    Class representing a decision tree classifier.
    """

    def __init__(self, min_sample_split: int = 2, max_depth: int = 10,
                 mode: Literal['gini', 'entropy'] = 'gini') -> None:
        """
        Creates a DecisionTreeClassifier object.

        Parameters
        ----------
        min_sample_split: int
            minimum number of samples required to split an internal node.
        max_depth: int
            maximum depth of the tree.
        mode: Literal['gini', 'entropy']
            the mode to use for calculating the information gain.
        """
        # parameters
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth  #levels of the tree
        self.mode = mode   #metric chosen to evaluate, normally an hiperparameter

        # estimated parameters
        self.tree = None
        self.dataset = None

    def _build_tree(self, dataset: Dataset, current_depth: int = 0) -> Node:
        """
        Builds a decision tree recursively.

        Parameters
        ----------
        dataset: Dataset
            The dataset to build the tree on.
        current_depth: int
            The current depth of the tree. (ponto de partida)

        Returns
        -------
        Node
            The root node of the tree.
        """
        n_samples = dataset.shape()[0]
        if n_samples >= self.min_sample_split and current_depth <= self.max_depth:
            #if the number of samples if higher or equal to the minimum split, split happens AND current depth is lower or equal to max depth (meaning that its not the last node)
            best_split = self._get_best_split(dataset)  #split happens
            if best_split['info_gain'] > 0:   
                #if info gain was 0 then we would be in the last leaf, but if it's higher than zero we have to choose branches
                left_subtree = self._build_tree(best_split['left_data'], current_depth + 1)  #does the same recursively
                right_subtree = self._build_tree(best_split['right_data'], current_depth + 1)
                return Node(best_split['feature_idx'], best_split['threshold'], left_subtree, right_subtree,
                            best_split['info_gain'])
        leaf_value = max(dataset.y, key=list(dataset.y).count)   #dataset.y is the data of the FINAL NODE
        #basically data just gets adjusted recursively and getting lower in the tree until one of the IF conditions don't happen
        return Node(value=leaf_value)



    def _get_best_split(self, dataset: Dataset) -> dict:
        """
        Finds the best split for a dataset based on the information gain.

        Parameters
        ----------
        dataset: Dataset
            The dataset to find the best split for.

        Returns
        -------
        dict
            A dictionary containing the best split containing the feature index, threshold, left and right datasets,
            and the information gain.
        """
        best_split = {}  #feature index, threshold, left and right datasets, and info gain
        info_gain = float('-inf')
        for feature_idx in range(dataset.shape()[1]):  #for each feature (shape()[1])
            features = dataset.X[:, feature_idx]  #each sample of the column (we are going column to column)
            possible_thresholds = np.unique(features)  #possible thresholds are the feature values for each sample
            for threshold in possible_thresholds[:-1]:
                left_data, right_data = self._split(dataset, feature_idx, threshold)  #split with possible thresholds
                y, left_y, right_y = dataset.y, left_data.y, right_data.y  #getting data for impurity
                current_info_gain = self._information_gain(y, left_y, right_y)  #y are data above and left and right the data after split
                if current_info_gain > info_gain:  #if current gain is better 
                    best_split = {
                        'feature_idx': feature_idx,
                        'threshold': threshold,
                        'left_data': left_data,
                        'right_data': right_data,
                        'info_gain': current_info_gain
                    }
                    info_gain = current_info_gain  #save for new iteration
        # check if best split is not empty (cases where labels of the dataset are all the same)
        if not best_split: 
            best_split = {'info_gain': info_gain}
        return best_split

    @staticmethod
    def _split(dataset: Dataset, feature_idx: int, threshold: float) -> Tuple[Dataset, Dataset]:
        """
        Splits a dataset into left and right datasets based on a feature and threshold.

        Parameters
        ----------
        dataset: Dataset
            The dataset to split.
        feature_idx: int
            The index of the feature to split on.
        threshold: float
            The threshold value to split on.

        Returns
        -------
        Tuple[Dataset, Dataset]
            A tuple containing the left and right datasets.
        """
        left_indices = np.argwhere(dataset.X[:, feature_idx] <= threshold).flatten()  #flatten goes from [[]] to []
        #indexes of samples with lower value than the threshold for the current feature
        right_indices = np.argwhere(dataset.X[:, feature_idx] > threshold).flatten()
        #indexes of samples with higher value than the threshold for the current feature
        left_data = Dataset(dataset.X[left_indices], dataset.y[left_indices], features=dataset.features,
                            label=dataset.label)
        right_data = Dataset(dataset.X[right_indices], dataset.y[right_indices], features=dataset.features,
                             label=dataset.label)
        return left_data, right_data

    def _information_gain(self, parent: np.ndarray, left_child: np.ndarray, right_child: np.ndarray) -> float:
        """
        Calculates the information gain of a split.
        It can be used for both gini and entropy.

        Parameters
        ----------
        parent: np.ndarray
            The parent data.
        left_child: np.ndarray
            The left child data.
        right_child: np.ndarray
            The right child data.

        Returns
        -------
        float
            The information gain of the split.
        """
        weight_left = len(left_child) / len(parent)  #weight of left split
        weight_right = len(right_child) / len(parent)  #weight of right split
        if self.mode == 'gini':
            return gini_impurity(parent) - (weight_left * gini_impurity(left_child) + weight_right * gini_impurity(right_child))
        elif self.mode == 'entropy':
            return entropy_impurity(parent) - (weight_left * entropy_impurity(left_child) + weight_right * entropy_impurity(right_child))
        else:
            raise ValueError(f'Invalid mode: {self.mode}. Valid modes are: "gini", "entropy"')

    def print_tree(self, tree: Node = None, indent: str = '\t') -> None:
        """
        Prints the decision tree.

        Parameters
        ----------
        tree: Node
            The root node of the tree.
        indent:
            The indentation to use.
        """
        if not tree:
            tree = self.tree
        if tree.value is not None:
            print(tree.value)
        else:
            print(f'{self.dataset.features[tree.feature_idx]} <= {tree.threshold}')
            print(f'{indent}left: ', end='')
            self.print_tree(tree.left, indent + '  ')
            print(f'{indent}right: ', end='')
            self.print_tree(tree.right, indent + '  ')

    def fit(self, dataset: Dataset) -> 'DecisionTreeClassifier':
        """
        Fits the decision tree classifier to a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to.

        Returns
        -------
        DecisionTreeClassifier
            The fitted model.
        """
        self.dataset = dataset
        self.tree = self._build_tree(dataset)
        return self

    def _make_prediction(self, x: np.ndarray, tree: Node) -> Union[float, str]:
        """
        Makes a prediction for a single sample.

        Parameters
        ----------
        x: np.ndarray
            The sample to make a prediction for.
        tree: Node
            The root node of the tree.

        Returns
        -------
        Union[float, str]
            The predicted value.
        """
        if tree.value is not None:  #if its not None means that the current node is a leaf and we return the leaf value
            return tree.value
        feature_value = x[tree.feature_idx]  #sample(index of the feature that the decision tree is splitting)
        if feature_value <= tree.threshold:  #as is lower or equal its left
            return self._make_prediction(x, tree.left)  #recursively calls the function with the sample x and left node as the new tree to explore until finds the leaf
        else:
            return self._make_prediction(x, tree.right)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Makes predictions for a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to make predictions for.

        Returns
        -------
        np.ndarray
            The predicted values.
        """
        predictions = [self._make_prediction(x, self.tree) for x in dataset.X]
        return np.array(predictions)

    def score(self, dataset: Dataset) -> float:
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

#Testing 
if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    data = read_csv(r'C:\Users\Fofinha\Desktop\UNI\MESTRADO\2o ANO\Sistemas Inteligentes\si\datasets\iris\iris.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = DecisionTreeClassifier(min_sample_split=3, max_depth=3, mode='gini')
    model.fit(train)
    model.print_tree()
    print(model.score(test))

# Review
# It will iterate through all the features and essentially ensure that the split made with a specific feature and a specific threshold is the one that provides the most gain.
# It does this as long as the split results in a gain, or until the number of samples and the depth allow it to.
# In case the split does not result in any gain, it reaches a leaf node.
# The tree is now constructed.
# We have the tree structure in place.
# Now, we take samples and run through the tree.
# We'll have an array of predictions that we should compare with the original data.
# We have the accuracy of the model
