from typing import Union
import numpy as np

from si.neural_networks.layers import Layer

class Dropout(Layer):
    """
    A randomset of neurons is temporarily ignored (dropped out) during training, helping prevent overfitting 
    by promoting robustness and generalization in the model.
    Some neurons are off - this is some values of X are multiply by zero
    """
    def __init__(self, probability: int):
        """
        Initialize the dropout layer.

        Parameters
        ----------
        probability: int
            The dropout rate, between 0 and 1.
            probability to desconet some connections
        
        """
        super().__init__()
        self.probability = probability #probability of being inactive
        
        #estimated parameters
        self.input = None
        self.output = None
        self.mask=None #matrix of 0 and 1

    def forward_propagation(self, input: np.ndarray, training: bool) -> np.ndarray:
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
        if training is True: #if we are training we do the dropout
            scaling_factor= 1-(1-self.probability)
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape) #binomial mask, values are 0 or 1, depending of being on or off
            self.output = input * self.mask * scaling_factor #some are being torned off because we are multiplying by zero, and the ones that are not turned off are multiplied by the scaling factor
            return self.output
        else: #if is test/inferece mode we can do dropout
            self.input = input
            return self.input

    def backward_propagation(self, output_error: np.ndarray) -> np.ndarray:
        """
        Perform backward propagation on the given output error.

        Parameters
        ----------
        output_error: np.ndarraay
            The output error of the layer.

        Returns
        -------
        ndarray
            The output error of the layer.
        """
        return self.mask * output_error #we get 3 neurons [0,1,0] and multiply by the array of error, and its expected that 2 are turned of, which we only do the backpropagaction in one neuron


    def output_shape(self) -> tuple:
        """
        Returns the output shape of the layer.

        Returns
        -------
        tuple
            The output shape of the layer.
        """
        return self._input_shape

    def parameters(self) -> int:
        """
        Returns the number of parameters of the layer.

        Returns
        -------
        int
            The number of parameters of the layer.
        """
        return 0 #no updated on weights or bias



#Testing 
if __name__ == '__main__':
    dropout_layer = Dropout(probability=0.5)

    input_data = np.random.rand(3, 4)  # Assuming 3 samples with 4 features each

    output_data = dropout_layer.forward_propagation(input_data, training=True)
    print("Forward Propagation (Training):\n", output_data)

    output_error = np.random.rand(*output_data.shape)

    backward_output_error = dropout_layer.backward_propagation(output_error)
    print("\nBackward Propagation:\n", backward_output_error)