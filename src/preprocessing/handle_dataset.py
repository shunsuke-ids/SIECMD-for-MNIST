import numpy as np


def shuffle_data(X, y):
    '''
    Shuffles Dataset (X, y)
    :param X: Data
    :param y: GT
    :return: Returns a randomly shuffled version of the Dataset
    '''
    assert X.shape[0] == y.shape[0]
    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)

    return X[idxs], y[idxs]


def split_ndata(X, y, n_train, n_test, n_val=0, shuffle=True):
    '''
    Splits Dataset into train, (val,) test
    :param X: Data
    :param y: GT
    :param n_train: Training portion (integer)
    :param n_test: Test portion (integer)
    :param n_val: Validation portion (integer, default=0)
    :param shuffle: Whether to shuffle the Dataset (default=True)
    :return: Either (X_train, X_val, X_test), (y_train, y_val, y_test) or
        (X_train, X_test), (y_train, y_test) depending on n_val being != 0
    '''
    assert len(X) == len(y) and len(X) >= n_train + n_test + n_val

    if shuffle:
        X, y = shuffle_data(X, y)

    idxs = np.arange(len(X))

    X_train, X_val, X_test = X[idxs[:n_train]], X[idxs[n_train:n_train + n_val]], X[
        idxs[n_train + n_val:n_train + n_val + n_test]]

    y_train, y_val, y_test = y[idxs[:n_train]], y[idxs[n_train:n_train + n_val]], y[
        idxs[n_train + n_val:n_train + n_val + n_test]]

    if n_val != 0:
        return (X_train, X_val, X_test), (y_train, y_val, y_test)
    else:
        return (X_train, X_test), (y_train, y_test)


def split_data(X, y, train_percent, test_percent, val_percent=0, shuffle=True):
    '''
    Split data into train, (val,) test, internally calls split_ndata
    :param X: Data
    :param y: GT
    :param train_percent: Training portion (real number in (0, 1])
    :param test_percent: Test portion (real number in (0, 1])
    :param val_percent: Validation portion (real number in (0, 1])
    :param shuffle: Whether to shuffle the Dataset (default=True)
    :return: Either (X_train, X_val, X_test), (y_train, y_val, y_test) or
        (X_train, X_test), (y_train, y_test) depending on n_val being != 0
    '''
    assert len(X) == len(y) and train_percent + test_percent + val_percent == 1

    n_train = int(train_percent * len(X))
    n_test = int(test_percent * len(X))
    n_val = int(val_percent * len(X))

    n_train = n_train + abs(n_train + n_test + n_val - len(X))

    return split_ndata(X, y, n_train, n_test, n_val, shuffle)


def normalDistribution(X, Y, n):
    '''
    Normal distribute Dataset (X, Y)
    :param X: Data
    :param Y: GT
    :param n: Prepares the dataset (X, Y) resulting in length n with equal representation of each class in Y
    :return: Processed Dataset (X, Y)
    '''
    n_classes = len(np.unique(Y))
    perClass = n / n_classes

    new_X, new_Y = np.zeros((n, *X.shape[1:])), np.zeros(n, dtype=np.int32)
    new_Y.fill(-1)

    i = 0
    for x, y in zip(X, Y):
        if i >= n: break
        if len(np.where(new_Y == y)[0]) < perClass:
            new_X[i], new_Y[i] = x, int(y)
            i += 1
    return new_X, new_Y

def make_3_channel_img(img):
    '''
    Returns grayscale image in RGB format
    :param img: image to convert
    :return: RGB format image
    '''
    return np.concatenate([img, img, img], axis=2)

def make_3_channel_imgs(imgs):
    '''
    Returns grayscale images in RGB format
    :param imgs: Array of images
    :return: RGB format images
    '''
    return np.array([make_3_channel_img(img) for img in imgs])
