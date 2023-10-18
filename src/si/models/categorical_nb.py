from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
import numpy as np

class CategoricalNB:
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize the Categorical Naive Bayes (suitable for categorical/discrete data)

        Parameters
        ----------
        smoothing: float
            Laplace smoothing to avoid zero probabilities
        """
        # Parameters
        self.smoothing = smoothing  # Parameter to avoid probabilities of 0

        # Attributes
        self.class_prior = None
        self.feature_prob = None
        self.n_classes = 0

    def fit(self, dataset: Dataset) -> 'CategoricalNB':
        """
        Fit the model to the given dataset.
        Estimate class_prior and feature_probs from the Dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: CategoricalNB
            The fitted model
        """
        n_samples, n_features = dataset.shape()
        labels = dataset.get_classes()
        n_classes = len(np.unique(labels))
        self.n_classes = n_classes
        self.n_features = n_features 

        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))  # Ensure correct shape
        class_prior = np.zeros(n_classes)

        for i in range(n_samples):
            sample = dataset.X[i]
            class_label = int(sample[-1])  # Class label is in the last column
            class_counts[class_label] += 1  # Counting how many times that class_label appears in the training set
            feature_counts[class_label, :] += sample # Counting the frequency of each value for each class

        class_counts += (n_features * self.smoothing)
        feature_counts += self.smoothing  # Laplace smoothing, avoiding probabilities of 0

        self.class_prior = class_counts / n_samples
        self.feature_prob = feature_counts / class_counts[:, np.newaxis]

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict the class labels for a given set of samples.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        class_probs = np.zeros(self.n_classes)  # Initialize an array to store class probabilities
        predictions = []

        for sample in dataset.X:
            for c in range(self.n_classes):
                prob = np.prod(sample * self.feature_prob[c] + (1 - sample) * (1 - self.feature_prob[c])) * self.class_prior[c]
                class_probs[c] = prob

            predicted_class = np.argmax(class_probs)
            predictions.append(predicted_class)

        return np.array(predictions)

    def score(self, dataset: Dataset) -> float:
        """
        Calculate the accuracy between actual values and predictions.

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        error: float
            Error between actual values and predictions
        """
        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)

if __name__ == '__main__':
    # import dataset
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    # initialize the CategoricalNB
    nb = CategoricalNB(smoothing=1.0)

    # fit the model to the train dataset
    nb.fit(dataset_train)

    prediction = nb.predict(dataset_test)

    # evaluate the model on the test dataset
    score = nb.score(dataset_test)
    print(f'The accuracy of the model is: {score}')