import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class VotingClassifier:
    """
    Ensemble classifier that uses the majority vote to predict the class labels.

    Parameters
    ----------
    models : array-like, shape = [n_models]
        Different models for the ensemble.

    Attributes
    ----------
    """
    def __init__(self, models):
        """
        Initialize the ensemble classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.

        """
        # parameters
        self.models = models

    def fit(self, dataset: Dataset) -> 'VotingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : VotingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)  #trains all models listed in the same dataset

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        y : array-like, shape = [n_samples]
            The predicted class labels.
        """

        # helper function
        def _get_majority_vote(pred: np.ndarray) -> int:
            """
            It returns the majority vote of the given predictions

            Parameters
            ----------
            pred: np.ndarray
                The predictions to get the majority vote of

            Returns
            -------
            majority_vote: int
                The majority vote of the given predictions
            """
            # get the most common label
            labels, counts = np.unique(pred, return_counts=True)  # an array of unique labels and an array of the counts of each unique label
            return labels[np.argmax(counts)]  #the label that corresponds to the maximum count in the counts array, which represents the majority vote

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()
        # iterates over each model in self.models, calls the predict method of each model with the input dataset, 
        # and stores the predictions in an array. The .transpose() method is then used to transpose the array, so each row corresponds to a sample, and each column corresponds to a model's prediction
        return np.apply_along_axis(_get_majority_vote, axis=1, arr=predictions)  #majority vote for each row

    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        dataset : Dataset
            The test data.

        Returns
        -------
        score : float
            Mean accuracy
        """
        return accuracy(dataset.y, self.predict(dataset))


if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    num_samples = 600
    num_features = 100
    num_classes = 2

    # random data
    X = np.random.rand(num_samples, num_features)  
    y = np.random.randint(0, num_classes, size=num_samples)  # classe aleatórios

    dataset_ = Dataset(X=X, y=y)

    #  features and class name
    dataset_.features = ["feature_" + str(i) for i in range(num_features)]
    dataset_.label = "class_label"
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)


    # initialize the KNN and Logistic classifier
    knn = KNNClassifier(k=3)
    lg = LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    # initialize the Voting classifier
    voting = VotingClassifier([knn, lg])

    voting.fit(dataset_train)

    # compute the score
    score = voting.score(dataset_test)
    print(f"Score: {score}")

    print(voting.predict(dataset_test))