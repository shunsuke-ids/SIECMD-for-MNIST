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


class VonMisesHead(nn.Module):
    """Von Mises分布を用いたクラス分類ヘッド

    クラスを単位円上に等間隔で固定配置し（μ_c = 2π * c / C）、
    CNNが出力した1次元スカラーzとの角度距離からクラスロジットを計算する。

    ロジット = cos(z - μ_c)
    これはVon Mises分布の尤度 p(x|c) ∝ exp(cos(z - μ_c)) の指数部に対応し、
    CrossEntropyLossに直接渡すことができる。
    κは全クラス共通のスカラーであり勾配スケールにしかならないため κ=1 に固定する。
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # μ_c: クラスを単位円上に等間隔配置（学習対象外）
        mu = torch.arange(num_classes, dtype=torch.float32) * (2.0 * np.pi / num_classes)
        self.register_buffer('mu', mu)

    def forward(self, z):
        """
        Args:
            z: CNNの出力スカラー (batch, 1)
        Returns:
            logits: クラスロジット (batch, num_classes)
        """
        # z: (batch, 1), self.mu: (num_classes,) → broadcasting で (batch, num_classes)
        logits = torch.cos(z - self.mu.unsqueeze(0))

        return logits


class VonMisesLearnedHead(nn.Module):
    """VonMisesHead のμ学習版

    クラスの角度間隔をsoftplusで正値化した累積和で表現し、
    周期的な順序構造を保ちながらμを学習する。

    ロジット = cos(z - μ_c)
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # raw_delta: 角度間隔の原パラメータ（学習可能）
        # softplus(raw_delta) が各クラス間の弧長になる
        # 初期値 0 → softplus(0) = log(2) ≈ 0.693 で等間隔に近い状態からスタート
        self.raw_delta = nn.Parameter(torch.zeros(num_classes)) # (num_classes,) 学習対象のパラメータとして登録

    def get_mu(self):
        delta = F.softplus(self.raw_delta)
        cumsum = torch.cumsum(delta, dim=0) # cumsum[i] = delta[0] + delta[1] + ... + delta[i]
        mu = 2.0 * np.pi * cumsum / cumsum[-1]
        return mu

    def forward(self, z):
        """
        Args:
            z: CNNの出力スカラー (batch, 1)
        Returns:
            logits: クラスロジット (batch, num_classes)
        """
        mu = self.get_mu()
        # z: (batch, 1), mu: (num_classes,) → broadcasting で (batch, num_classes)
        logits = torch.cos(z - mu.unsqueeze(0))

        return logits