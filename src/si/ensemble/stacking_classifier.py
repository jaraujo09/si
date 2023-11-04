from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.models import knn_classifier, logistic_regression, decision_tree_classifier
from si.model_selection.split import train_test_split
import numpy as np

class StackingClassifier:
    """
    Stacking is an ensemble machine learning technique that combines multiple base classifiers or models to improve
    prediction accuracy. It is a form of model averaging, where the predictions of multiple models are combined to 
    make a final prediction
    """
    def __init__(self, models, final_model):
        """
        Initialize the ensemble stacking classifier.

        Parameters
        ----------
        models: array-like, shape = [n_models]
            Different models for the ensemble.
        fina_model:str 
            the model to make the final predictions
        """
        self.models = models  #used models
        self.final_model = final_model #final to make predictions

    def fit(self, dataset:Dataset)->'StackingClassifier':
        """
        Fit the models according to the given training data.

        Parameters
        ----------
        dataset : Dataset
            The training data.

        Returns
        -------
        self : StackingClassifier
            The fitted model.
        """
        for model in self.models:
            model.fit(dataset)
        
        preds = []
        for model in self.models:
            predic = model.predict(dataset)
            preds.append(predic)

        #right now I have an array were rows are models and columns are values of y for each sample - so I'm going to transpose
        predictions = np.array(preds).T
        self.final_model.fit(Dataset(dataset.X, predictions))

        return self
    
    def predict(self, dataset:Dataset)->np.ndarray:
        """
        Collects the predictions of all the models and computes the final prediction of the final model returning it.
        Args:
            dataset (Dataset): Dataset 
        Returns:
            np.array: Final model prediction
        """
        # gets the model predictions
        predictions = []
        for model in self.models:
            predictions.append(model.predict(dataset))

        # gets the final model previsions
        y_pred = self.final_model.predict(Dataset(np.array(predictions).T, dataset.y))

        return y_pred

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
        y_pred = self.predict(dataset)
        score = accuracy(dataset.y, y_pred)

        return score


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import stratified_train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    filename_breast = r"C:\Users\Fofinha\Desktop\UNI\MESTRADO\2o ANO\Sistemas Inteligentes\si\datasets\breast_bin\breast-bin.csv"
    breast=read_csv(filename_breast, sep=",",features=True,label=True)
    train_data, test_data = stratified_train_test_split(breast, test_size=0.20, random_state=42)

    #knnregressor
    knn = KNNClassifier(k=3)
    
    #logistic regression
    LG=LogisticRegression(l2_penalty=1, alpha=0.001, max_iter=1000)

    #decisiontreee
    DT=DecisionTreeClassifier(min_sample_split=3, max_depth=3, mode='gini')

    #final model
    final_modelo=knn
    modelos=[knn,LG,DT]
    exercise=StackingClassifier(modelos,final_modelo)
    exercise.fit(train_data)
    print(exercise.score(test_data))