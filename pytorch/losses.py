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
    
class EuclideanVectorLoss(nn.Module):
    """
    SVLのEuclideanバージョン
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
        # 逆伝播時に sqrt の勾配は 1/(2*sqrt(x)) の形になるため、x が0に近いと数値的に不安定になりNaNが発生する
        # そのため、微小な値を足して安定化を図る
        euclidean_dist = torch.sqrt(squared_diff + 1e-8)
        loss = euclidean_dist.mean()

        return loss

class ArcDistanceVectorLoss(nn.Module):
    """
    予測ベクトルを単位円上に正規化し、
    正解クラスベクトルとの円弧距離（ラジアン）を損失とする
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
        dot_product = torch.clamp(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
        loss = torch.acos(dot_product)

        return loss.mean()

class CircularSoftLabelCrossEntropyLoss(nn.Module):
    """正解クラス0.8・隣接クラス各0.1のソフトラベルCross Entropy

    クラスが循環的に並ぶ場合に適したソフトラベル。
    隣接は (c-1) % C と (c+1) % C に割り当てる。

    Parameters:
        num_classes (int): クラス数
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, y_true):
        """
        Args:
            logits: モデル出力 (batch, num_classes)
            y_true: sparse正解ラベル (batch,)
        Returns:
            スカラー損失
        """
        C = self.num_classes
        batch_size = logits.size(0)

        # ソフトラベルを構築
        soft_labels = torch.zeros(batch_size, C, device=logits.device)
        soft_labels.scatter_(1, y_true.long().unsqueeze(1), 0.8)
        prev_cls = (y_true.long() - 1) % C
        next_cls = (y_true.long() + 1) % C
        soft_labels.scatter_add_(1, prev_cls.unsqueeze(1), torch.full((batch_size, 1), 0.1, device=logits.device))
        soft_labels.scatter_add_(1, next_cls.unsqueeze(1), torch.full((batch_size, 1), 0.1, device=logits.device))

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_labels * log_probs).sum(dim=1).mean()
        return loss


class ExpectedCircularDistanceLoss(nn.Module):
    """期待循環距離損失

    予測分布の下での期待循環距離を最小化する。
    評価指標 circular MAE を直接最小化する損失。

    L = Σ_c softmax(logit_c) * d_circ(c, y_true) / (C/2)

    d_circ(c, y) = min(|c - y|, C - |c - y|)  ← [0, C/2] の循環距離
    正規化して [0, 1] にスケール。

    Parameters:
        num_classes (int): クラス数
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        # 各クラスペア間の循環距離テーブルを事前計算 (C, C)
        C = num_classes
        idx = torch.arange(C)
        # dist_table[i, j] = d_circ(i, j) / (C/2) ∈ [0, 1]
        diff = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()  # (C, C)
        circ_dist = torch.minimum(diff, C - diff).float() / (C / 2)
        self.register_buffer('dist_table', circ_dist)

    def forward(self, logits, y_true):
        """
        Args:
            logits: モデル出力 (batch, num_classes)
            y_true: sparse正解ラベル (batch,)
        Returns:
            スカラー損失
        """
        probs = F.softmax(logits, dim=1)  # (batch, C)
        # 正解クラスごとの循環距離ベクトルを取得 (batch, C)
        dists = self.dist_table[y_true.long()]
        loss = (probs * dists).sum(dim=1).mean()
        return loss


class VonMisesSoftLabelCELoss(nn.Module):
    """Von Mises分布によるソフトラベルCross Entropy損失

    クラスを単位円上に等間隔固定配置し、Von Mises分布で
    正解クラス周辺のソフトラベルを生成してCE損失を計算する。

    target_c = softmax(κ · cos(2π(c − y) / C))

    κ→∞ で通常CE（one-hot）、κ→0 で均一ラベルに近づく。

    Parameters:
        num_classes (int): クラス数
        kappa (float): 集中度パラメータ（初期値）
        learn_kappa (bool): κを学習可能にするか（default: False）
    """
    def __init__(self, num_classes, kappa=1.0, learn_kappa=False):
        super().__init__()

        # cos_diffs[i, j] = cos(2π(i-j)/C) ← Von Misesの指数部
        C = num_classes
        angles = torch.arange(C, dtype=torch.float32) * (2.0 * np.pi / C)
        cos_diffs = torch.cos(angles.unsqueeze(0) - angles.unsqueeze(1))  # (C, C)
        self.register_buffer('cos_diffs', cos_diffs)

        # κは正値を保証するため log_κ で保持
        log_kappa = torch.tensor(float(np.log(kappa)))
        if learn_kappa:
            self.log_kappa = nn.Parameter(log_kappa)
        else:
            self.register_buffer('log_kappa', log_kappa)

    def get_kappa(self):
        return torch.exp(self.log_kappa)

    def forward(self, logits, y_true):
        """
        Args:
            logits: モデル出力 (batch, num_classes)
            y_true: sparse正解ラベル (batch,)
        Returns:
            スカラー損失
        """
        kappa = self.get_kappa()
        # 正解クラスに対応する cos行を取得 (batch, C)
        cos_vals = self.cos_diffs[y_true.long()]
        # Von Misesソフトラベル (batch, C)
        soft_targets = F.softmax(kappa * cos_vals, dim=1)

        log_probs = F.log_softmax(logits, dim=1)
        loss = -(soft_targets * log_probs).sum(dim=1).mean()
        return loss


class CombinedCEMSEVectorLoss(nn.Module):
    """CrossEntropyLoss + λ * MSEVectorLoss の線形結合損失

    L = CE(logits, y) + λ * MSEVectorLoss(logits, y)

    Parameters:
        num_classes (int): クラス数
        lambda_circ (float): 円環ロスの重み λ (default: 1.0)
    """
    def __init__(self, num_classes, lambda_circ=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.msevl = MSEVectorLoss(num_classes)
        self.lambda_circ = lambda_circ

    def forward(self, logits, y_true):
        ce_loss = self.ce(logits, y_true)
        mse_loss = self.msevl(logits, y_true)
        return ce_loss + self.lambda_circ * mse_loss