import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:
    """
    The RidgeRegression is a linear model using the L2 regularization.
    This model solves the linear regression problem using an adapted Least Squares technique

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    mean : np.ndarray
        Mean of the dataset for every feature
    std : np.ndarray
        Standard deviation of the dataset for every feature
    """

    def __init__(self, l2_penalty: float, scale: bool):
        """
        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter
        scale: bool
            Whether to scale the dataset or not
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None  #y intercept
        self.mean = None
        self.std = None
    
    def fit(self, dataset : Dataset) -> 'RidgeRegressionLeastSquares':
        """
        Fit the model to the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to

        Returns
        -------
        self: RidgeRegressionLeastSquares
            The fitted model
        """
        if self.scale:  #in case its necessary to scale
            # compute mean and std
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            # scale the dataset
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X
        
        m, n = dataset.shape()

        X = np.c_[np.ones(m), X]  #add 1 column of ones in the first column position (intercept term)

        penalty_matrix = self.l2_penalty*np.eye(n+1)  #add +1 because I added one more feature above
        penalty_matrix[0,0] = 0 #the first position must be 0 because of theta zero

        #Model parameters

        transposed_X = X.T

        XTX = np.linalg.inv(transposed_X.dot(X) + penalty_matrix)

        XTy = transposed_X.dot(dataset.y)

        thetas=XTX.dot(XTy)
        self.theta_zero=thetas[0]
        self.theta=thetas[1:] #theta (remaining elements)
        
        return self
    
    def predict(self, dataset : Dataset)->np.ndarray:
        """
        Predict the output of the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        predictions: np.array
            The predictions of the dataset
        """
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        m,n = dataset.shape()
        X = np.c_[np.ones(m), X]  #add 1 column of ones in the first column position (intercept term)
        predic = X.dot(np.r_[self.theta_zero, self.theta]).flatten() #concatenate t0 and theta and multiplies the matrices

        return predic
    
    def score(self, dataset : Dataset)->float:
        """
        Compute the Mean Square Error of the model on the dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the MSE on

        Returns
        -------
        mse: float
            The Mean Square Error of the model
        """
        y_pred = self.predict(dataset)

        return mse(dataset.y, y_pred)



# This is how you can test it against sklearn to check if everything is fine
if __name__ == '__main__':
    # make a linear dataset
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([2, 3, 4, 5])
    dataset_ = Dataset(X, y)
    print('My model')
    # fit the model
    model = RidgeRegressionLeastSquares(l2_penalty=0.1, scale = True)
    model.fit(dataset_)
    print(model.theta)
    print(model.theta_zero)

    # compute the score
    print(model.score(dataset_))
    print('\n Compare with sklearn')
    # compare with sklearn
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=2.0)
    # scale data
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(model.coef_) # should be the same as theta
    print(model.intercept_) # should be the same as theta_zero
    print(mse(dataset_.y, model.predict(X)))
