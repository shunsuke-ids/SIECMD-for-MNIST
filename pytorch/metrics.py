import numpy as np

def soft_confusion_matrix(softmax_probs, y_true, num_classes):
    """
    クラスごとの平均Softmax分布（ソフト混同行列）

    soft_cm[i, j] = 真値クラスiのサンプルに対するクラスjへの平均Softmax確率
    通常の混同行列と異なり、モデルが「どこに確率質量を置いているか」を示す。

    Args:
        softmax_probs: Softmax確率 (N, num_classes)
        y_true: 正解ラベル (N,)
        num_classes: クラス数
    Returns:
        ndarray: (num_classes, num_classes)
    """
    soft_cm = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        mask = y_true == i # 真値クラスiのサンプルを選択
        if mask.sum() > 0:
            soft_cm[i] = softmax_probs[mask].mean(axis=0)
    return soft_cm


def circular_mae(y_pred, y_true, num_classes):
    """
    周期性を考慮したMAE（Mean Absolute Error）
    """
    diff = np.abs(y_pred - y_true)
    circular_diff = np.minimum(diff, num_classes - diff)

    return circular_diff.mean()


def circular_mae_per_class(y_pred, y_true, num_classes):
    """
    クラスごとの周期性を考慮したMAE（Mean Absolute Error）
    クラス不均衡に対応するため、各クラスで計算して平均を取る

    Returns:
        dict: {
            'per_class': {class_idx: cMAE, ...},  # 各クラスのcMAE
            'macro': float  # マクロ平均cMAE
        }
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    diff = np.abs(y_pred - y_true)
    circular_diff = np.minimum(diff, num_classes - diff)

    per_class_cmae = {}
    for class_idx in range(num_classes):
        mask = y_true == class_idx
        if mask.sum() > 0:
            per_class_cmae[class_idx] = circular_diff[mask].mean()
        else:
            per_class_cmae[class_idx] = np.nan  # サンプルがないクラス

    # マクロ平均（nanを除外して計算）
    valid_cmaes = [v for v in per_class_cmae.values() if not np.isnan(v)]
    macro_cmae = np.mean(valid_cmaes) if valid_cmaes else np.nan

    return {
        'per_class': per_class_cmae,
        'macro': macro_cmae
    }