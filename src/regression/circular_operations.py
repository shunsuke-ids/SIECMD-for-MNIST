import numpy as np


def eigen_median(points):
    '''
    Compute the Median of some 2D vectors (points) with length 1
    Comparison of Methods for Hyperspherical Data Averaging and Parameter Estimation (K. Rothaus et al.)
    :param points: Array of 2D vectors (points)
    :return: Median value of input points
    '''
    n = len(points)
    d = len(points[0])
    M = np.zeros((n, d, d))
    for i in range(n):
        M[i] = np.outer(points[i], points[i])
    M = np.sum(M, axis=0)
    M = M / n
    eigvals, eigvecs = np.linalg.eig(M)

    mean = eigvecs[:, np.argmax(eigvals)]
    d0 = np.sum([np.linalg.norm(mean - points[i]) for i in range(n)])
    d1 = np.sum([np.linalg.norm(-mean - points[i]) for i in range(n)])
    return mean if d0 < d1 else -mean


def _max_difference(arr):
    '''
    Helper function for circular_mean. Calculates the difference between the highest and lowest value in arr
    :param arr: Input array
    :return: Difference
    '''
    arr = np.sort(arr)
    return arr[-1] - arr[0]


def circular_mean(arr, period=360):
    '''
    Calculates the mean of an array of circular values.
    :param arr: Array of circular data
    :param period: Period length
    :return: Calculated mean
    '''
    arr = np.array(arr) if isinstance(arr, list) else arr
    arr = np.sort(arr)

    _max_diff = _max_difference(arr)
    _arr = arr.copy()
    for j in range(len(arr)):
        arr[j] = arr[j] + period
        max_diff = _max_difference(arr)
        if max_diff < _max_diff:
            _max_diff = max_diff
            _arr = arr.copy()
    mean = np.mean(_arr)
    mean = mean % period
    return mean
