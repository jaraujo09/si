import numpy as np


def sigmoid_function(X: np.ndarray) -> np.ndarray:
    """
    It returns the sigmoid function of the given input

    Parameters
    ----------
    X: np.ndarray
        The input of the sigmoid function

    Returns
    -------
    sigmoid: np.ndarray
        The sigmoid function of the given input
    """
    return 1 / (1 + np.exp(-X))


if __name__ == "__main__":
    # Test data
    input_data = np.array([0, 1, 2, 3, 4])

    # Test sigmoid_function
    sigmoid_result = sigmoid_function(input_data)
    print(f"Sigmoid Function for input_data: {sigmoid_result}")
