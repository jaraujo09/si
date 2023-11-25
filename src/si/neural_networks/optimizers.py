from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None  #accumulated gradient from previous epochs (iteractions)

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient
    

class Adam(Optimizer):

    def __init__(self, learning_rate: float = 0.01, beta_1:float=0.9,beta_2:float=0.999,epsilon: float = 1e-8):
        """
        Initialize the optimizer.
        Combination of RMSprop and SGD with momentum

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        beta_1:float
            The exponential decay rate for the 1st moment estimates
        beta_2:float
            The exponential decay rate for the 2nd moment estimates
        epsilon:float
            A small constantfor numerical stability
        """
        super().__init__(learning_rate) #gets learning rate parameter
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        #estimated parameters
        self.m = None #moving average m
        self.v = None #moving average v
        self.t = 0 #time stamp (epoch), initialized as 0

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray #perda associada aos pesos
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.m is None and self.v is None:
            self.m = np.zeros(np.shape(w))
            self.v= np.zeros(np.shape(w))
        else:
            self.t +=1
            
            self.m=self.beta_1*self.m + ((1-self.beta_1)*grad_loss_w)
            self.v=self.beta_2*self.v + ((1-self.beta_2)*(grad_loss_w**2))

            # bias 
            m_hat=self.m/(1-self.beta_1**self.t)
            v_hat=self.v/(1-self.beta_2**self.t)

            w= w - (self.learning_rate*(m_hat/(np.sqrt(v_hat) + self.epsilon)))
        
        return w 
        
