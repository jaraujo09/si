import numpy as np

def manhattan_distance (x: np.ndarray, y: np.ndarray)-> np.ndarray:
    """
    It computes the manhattan distance of a point (x) to a set of points y.
        distance_x_y1 = |x1 - y11| + |x2 - y12| + ... + |xn - y1n|
        distance_x_y2 = |x1 - y21| + |x2 - y22| + ... + |xn - y2n|
        ...

    Parameters
    ----------
    x: np.ndarray
        Point (a single sample)
    y: np.ndarray
        Set of points (multiple samples).

    Returns
    -------
    np.ndarray
        Manhattan distance between X and the various samples in y.
    """
    
    return np.abs((x - y).sum(axis = 1))


if __name__ == '__main__':
    # test manhattan_distance
    x = np.array([1, 2, 3])
    y = np.array([[1, 2, 3], [4, 5, 6]])
    our_distance = manhattan_distance(x, y)
    # using sklearn
    # to test this snippet, you need to install sklearn (pip install -U scikit-learn)
    from sklearn.metrics.pairwise import manhattan_distances
    sklearn_distance = manhattan_distances(x.reshape(1, -1), y)
    assert np.allclose(our_distance, sklearn_distance)
    print(our_distance, sklearn_distance)