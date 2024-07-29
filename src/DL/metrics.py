import numpy as np

def accuracy_angles(y_true, y_pred, prediction_tolerance=45):
    '''
    Returns standard classification accuracy metric for predicted angles with a tolerance
    :param y_true: GT angles
    :param y_pred: Predicted angles
    :param prediction_tolerance: Classification tolerance (default 45 degree)
    :return: Classification accuracy in [0, 1]
    '''
    t = 0
    for i, prediction in enumerate(y_pred):
        if prediction <= y_true[i] + prediction_tolerance and prediction >= y_true[i] - prediction_tolerance:
            t += 1
    return t / len(y_pred)


def accuracy_angles_areas(y_true, y_pred, n_areas=4):
    '''
    Returns standard classification accuracy metric. For this angles are transfomed into areas (classes).
    :param y_true: GT angles
    :param y_pred: Predicted angles
    :param n_areas: Number of areas (classes) default=4 (quadrants)
    :return: Classification accuracy in [0, 1]
    '''
    area_size = 360 / n_areas
    if area_size != int(area_size): return 0

    t = 0
    for i, pred in enumerate(y_pred):
        tolerance_range = (y_true[i] - (y_true[i] % area_size)) / area_size
        if tolerance_range * area_size <= pred and pred < (tolerance_range + 1) * area_size:
            t += 1
    return t / len(y_pred)


def prediction_mean_deviation(y_true, y_pred):
    '''
    Returns mean deviation for predicted angles, as an accuracy metric for regression task
    :param y_true: GT angles
    :param y_pred: Predicted angles
    :return: $ \mathbf{E}_{deg} = \frac{1}{n} \sum_{i=1}^n min\left(|\alpha_i - \beta_i|, 2\pi - |\alpha_i - \beta_i|\right) $
    '''
    diviations = np.zeros(y_true.shape, dtype=np.uint16)
    for i, pred in enumerate(y_pred):
        diviations[i] = min(abs(y_true[i] - pred), 360 - abs(y_true[i] - pred))
    return np.mean(diviations)

def get_mean_deviations_for_n_areas(n_areas):
    '''
    Returns a matrix (n_areas, n_areas) of mean deviations, which are calculated through min and max deviation in areas.
    Rows represent the predicted bin and columns represent the GT.
    :param n_areas: Number of areas (classes, sections or bins)
    :return: matrix (n_areas, n_areas) with mean deviations.
    '''
    area_size = 360 / n_areas
    correct_deviation = (area_size) / 4

    deviations = np.zeros((n_areas, n_areas))
    deviations[0][0] = correct_deviation
    for i in range(1, n_areas):
        true = np.arange(0, area_size + 1)
        pred = i * area_size + int(area_size / 2)

        max_deviation, min_deviation = 0, np.inf
        for j in range(int(area_size + 1)):

            deviation = min(abs(true[j] - pred), 360 - abs(true[j] - pred))
            max_deviation = max(max_deviation, deviation)
            min_deviation = min(min_deviation, deviation)
        deviations[0][i] = (max_deviation + min_deviation) / 2
    for i in range(1, n_areas):
        for j in range(n_areas):
            deviations[i][j] = deviations[0][(j - i) % n_areas]
    return deviations


def mean_deviation_areas(y_true, y_pred, n_areas):
    '''
    Returns mean angle deviation for predicted bins, with n_areas equal to the number of bins
    :param y_true: GT bin
    :param y_pred: Predicted bin
    :param n_areas: Number of areas (classes, bins)
    :return: \mathbf{E}_{deg} \ = \ \frac{1}{n} \sum_{i=1}^n \frac{min\_deviation_i + max\_deviation_i}{2}
    '''
    devs = np.zeros(len(y_pred))
    deviations = get_mean_deviations_for_n_areas(n_areas)
    n_samples = len(y_true)
    deviation, n_correct = 0, 0
    for i in range(n_samples):
        if y_true[i] == y_pred[i]: n_correct += 1
        deviation += deviations[y_true[i]][y_pred[i]]
        devs[i] = deviations[y_true[i]][y_pred[i]]
    mean_deviation = deviation / n_samples
    accuracy = n_correct / n_samples
    return mean_deviation, accuracy, devs
