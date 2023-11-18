import copy
from abc import abstractmethod

import numpy as np

from si.neural_networks.optimizers import Optimizer


class Layer:
    """
    Base class for neural network layers.
    """

    @abstractmethod # allows that all methods from this class are used by the class that used, in this case, the Layer class
    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
        """
        Perform forward propagation on the given input, i.e., computes the output of a layer for a given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, output_error: float) -> float:
        """
        Perform backward propagation on the given output error, i.e., computes dE/dX for a given dE/dY and update
        parameters if any.

        Parameters
        ----------
        output_error: float
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        raise NotImplementedError

    def layer_name(self) -> str:
        """
        Returns the name of the layer.

        Returns
        -------
        str
            The name of the layer.
        """
        return self.__class__.__name__  #gets the name of the class

    @abstractmethod
    def output_shape(self) -> tuple:
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        raise NotImplementedError

    def set_input_shape(self, shape: tuple):
        """
        Sets the shape of the input to the layer.

        Parameters
        ----------
        shape: tuple
            The shape of the input to the layer.
        """
        self._input_shape = shape

    def input_shape(self) -> tuple:
        """
        Returns the shape of the input to the layer.

        Returns
        -------
        tuple
            The shape of the input to the layer.
        """
        return self._input_shape

    @abstractmethod
    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        raise NotImplementedError


class DenseLayer(Layer):  #gets the abstract of Layer
    """
    Dense layer of a neural network.
    """

    def __init__(self, n_units: int, input_shape: tuple = None):
        """
        Initialize the dense layer.

        Parameters
        ----------
        n_units: int
            The number of units of the layer, aka the number of neurons, aka the dimensionality of the output space.
        input_shape: tuple
            The shape of the input to the layer.
        """
        super().__init__()
        self.n_units = n_units  #nr of neurons
        self._input_shape = input_shape  #nr of features

        self.input = None
        self.output = None
        self.weights = None  #connect with the next layer
        self.biases = None  #each neuron has an associated bias


    def initialize(self, optimizer: Optimizer) -> 'DenseLayer':  #get random weights and bias that will later be optimized
        # initialize weights from a 0 centered uniform distribution [-0.5, 0.5)
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5  #rows = features and colums = neurons, which means features*neurons = weights
        
        # initialize biases to 0
        self.biases = np.zeros((1, self.n_units))  #number of zeros are the size of neurons, bc each neuron has a bias
        self.w_opt = copy.deepcopy(optimizer) 
        self.b_opt = copy.deepcopy(optimizer)
        return self

    def parameters(self) -> int:  #weights(rows*columns) + bias
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)  #np.prod does multiplication

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:  #bool to know if we're training or not
        """
        Perform forward propagation on the given input.

        Parameters
        ----------
        input: numpy.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or in inference mode.

        Returns
        -------
        numpy.ndarray
            The output of the layer.
        """
        self.input = input  # X in the firts layer
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error: np.ndarray) -> float:
        """
        Perform backward propagation on the given output error.
        Computes the dE/dW, dE/dB for a given output_error=dE/dY.
        Returns input_error=dE/dX to feed the previous layer.

        Parameters
        ----------
        output_error: numpy.ndarray
            The output error of the layer.

        Returns
        -------
        float
            The input error of the layer.
        """
        # computes the layer input error (the output error from the previous layer),
        # dE/dX, to pass on to the previous layer
        input_error = np.dot(output_error, self.weights.T)  #error to discover to go to the layer before, knowing we have the error output as input
        
        # computes the weight error: dE/dW = X.T * dE/dY
        #error associated to the weights. has to be equal to the number os existing weights in the layers before and after
        weights_error = np.dot(self.input.T, output_error)
        
        # computes the bias error: dE/dB = dE/dY
        #equals to the number os neurons in the last layer
        #np.dot(A,B) = ncolsA = nrowsB
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        # updates parameters
        self.weights = self.w_opt.update(self.weights, weights_error)  #optimization and update
        self.biases = self.b_opt.update(self.biases, bias_error)   #we can use gradient discent as other optimizers
        return input_error


    def output_shape(self) -> tuple:  #to be tabular
        """
        Returns the shape of the output of the layer.

        Returns
        -------
        tuple
            The shape of the output of the layer.
        """
        return (self.n_units,)  #number of neurons
