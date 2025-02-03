# Function to compute standard deviation
def my_std(x):
    """_summary_

    Parameters
    ----------
    x : Sequence of Numbers

    Returns
    -------
    std: Float
        Standard deviation of the x
    """
    import math  # import math module

    N = len(x)  # number of elements in
    mean = sum(x) / N  # mean of the data
    S = sum(x**2 for x in N) / N  # mean of squares
    Var = S - mean**2  # variance
    std = math.sqrt(Var)  # standard deviation

    return std


# Function that returns the position of the largest element in a given sequence (list, tuple, array etc.)
def my_argmax(x):
    """Returns the index of the largest element in the sequence x.

    Parameters
    ----------
    x : Sequence of Numbers

    Returns
    -------
    i : Int
        Index of the largest element in the sequence x
    """
    max_val = x[0]
    max_index = 0
    for i, val in enumerate(x):
        if val > max_val:
            max_val = val
            max_index = i
    return max_index
