import numy as np

def circular_mae(y_pred, y_true, num_classes):
    """
    周期性を考慮したMAE（Mean Absolute Error）
    """
    diff = np.abs(y_pred - y_true)
    circular_diff = np.minimum(diff, num_classes - diff)

    return circular_diff.mean()