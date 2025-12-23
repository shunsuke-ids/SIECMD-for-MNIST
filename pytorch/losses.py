import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SoftmaxVectorLoss(nn.Module): # nn.Moduleクラスを継承
    """
    クラスを単位円上に配置し、Softmax出力の重み付き合成ベクトルと
    真値ベクトルとの内積ベースの損失を計算
    
    Parameters:
        num_classes (int): クラス数
    """
    def __init__(self, num_classes): # 最初に自動で呼び出されるメソッド
        super().__init__() # 親クラスの初期化
        self.num_classes = num_classes

        angles = torch.arange(num_classes, dtype=torch.float32) * (2.0 * np.pi / num_classes)
        class_coords = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        self.register_buffer('class_coords', class_coords)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: Softmax出力 (batch_size, num_classes)
            y_true: sparse形式の正解ラベル (batch_size,)
        
        Returns:
            ロス
        """
        y_pred_softmax = F.softmax(y_pred, dim=1)
        y_true_onehot = F.one_hot(y_true.long(), num_classes=self.num_classes).float().to(y_pred.device)

        pred_vector = torch.matmul(y_pred_softmax, self.class_coords)
        true_vector = torch.matmul(y_true_onehot, self.class_coords)

        dot_product = torch.sum(pred_vector * true_vector, dim=-1)
        loss = 1.0 - dot_product

        return loss.mean()

class NormalizedSoftmaxVectorLoss(nn.Module):
    """
    SoftmaxVectorLossを正規化したバージョン
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        angles = torch.arange(num_classes, dtype=torch.float32) * (2.0 * np.pi / num_classes)
        class_coords = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        self.register_buffer('class_coords', class_coords)

    def forward(self, y_pred, y_true):
        y_pred_softmax = F.softmax(y_pred, dim=1)
        y_true_onehot = F.one_hot(y_true.long(), num_classes=self.num_classes).float().to(y_pred.device)

        pred_vector = torch.matmul(y_pred_softmax, self.class_coords)
        true_vector = torch.matmul(y_true_onehot, self.class_coords)

        pred_vector = F.normalize(pred_vector, p=2, dim=-1)

        dot_product = torch.sum(pred_vector * true_vector, dim=-1)
        loss = 1.0 - dot_product

        return loss.mean()
    
class MSEVectorLoss(nn.Module):
    """
    SVLのMSEバージョン
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        angles = torch.arange(num_classes, dtype=torch.float32) * (2.0 * np.pi / num_classes)
        class_coords = torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)
        self.register_buffer('class_coords', class_coords)

    def forward(self, y_pred, y_true):
        y_pred_softmax = F.softmax(y_pred, dim=1)
        y_true_onehot = F.one_hot(y_true.long(), num_classes=self.num_classes).float().to(y_pred.device)

        pred_vector = torch.matmul(y_pred_softmax, self.class_coords)
        true_vector = torch.matmul(y_true_onehot, self.class_coords)

        squared_diff = torch.sum((pred_vector - true_vector) ** 2, dim=-1)
        loss = squared_diff.mean()

        return loss