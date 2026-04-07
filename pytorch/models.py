import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, image_size=28):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5) # フィルター数が16, カーネルサイズが5
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2,2)

        self._calculate_flatten_size(input_channels, image_size) # フラット化後の特徴量のサイズを計算

        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc3 = nn.Linear(16, num_classes)

        self.dropout = nn.Dropout(0.5)

    def _calculate_flatten_size(self, input_channels, image_size): # 内部で使用するメソッドはアンダースコアで始める慣習
        with torch.no_grad(): # 勾配計算を無効化
            x = torch.zeros(1, input_channels, image_size, image_size) # ダミー入力
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            self.flatten_size = x.view(1, -1).size(1) # 一次元化後のサイズ, インスタンス変数self.flatten_sizeに保存

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        x = x.view(x.size(0), -1) # バッチサイズを維持しつつ一次元化

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
class ResNet18(nn.Module):
    def __init__(self, input_channels=1, num_classes=10, image_size=224):
        super().__init__()
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        self.model = model

    def forward(self, x):
        return self.model(x)

class VonMisesModel(nn.Module):
    """SimpleCNN（スカラー出力）+ VonMisesHead の組み合わせモデル

    SimpleCNNをnum_classes=1で使いバックボーンを共有する。
    出力形状は (batch, num_classes) で通常のCrossEntropyLossと互換。
    """
    def __init__(self, input_channels=1, num_classes=10, image_size=28, arch='simple_cnn'):
        super().__init__()
        if arch == 'simple_cnn':
            self.backbone = SimpleCNN(input_channels, num_classes=1, image_size=image_size)
        elif arch == 'resnet18':
            self.backbone = ResNet18(input_channels, num_classes=1, image_size=image_size)
        self.von_mises_head = VonMisesHead(num_classes)

    def forward(self, x):
        z = self.backbone(x)                    # (batch, 1)
        logits = self.von_mises_head(z)         # (batch, num_classes)
        return logits


class VonMisesLearnedModel(nn.Module):
    """SimpleCNN（スカラー出力）+ VonMisesLearnedHead の組み合わせモデル

    VonMisesModelと同一だが、μを学習可能なVonMisesLearnedHeadを使用する。
    """
    def __init__(self, input_channels=1, num_classes=10, image_size=28, arch='simple_cnn'):
        super().__init__()
        if arch == 'simple_cnn':
            self.backbone = SimpleCNN(input_channels, num_classes=1, image_size=image_size)
        elif arch == 'resnet18':
            self.backbone = ResNet18(input_channels, num_classes=1, image_size=image_size)
        self.von_mises_head = VonMisesLearnedHead(num_classes)

    def forward(self, x):
        z = self.backbone(x)                    # (batch, 1)
        logits = self.von_mises_head(z)         # (batch, num_classes)
        return logits
    
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

        # raw_delta: クラス間角度間隔の原パラメータ（学習可能）
        # C個のギャップ全てを学習し、mu[0]=0 固定で回転の自由度を除去
        # 初期値 0 → softplus(0) = log(2) で全ギャップ等値 → 等間隔スタート
        self.raw_delta = nn.Parameter(torch.zeros(num_classes))

    def get_mu(self):
        delta = F.softplus(self.raw_delta)              # (C,) 全ギャップ
        total = delta.sum()
        cumsum = torch.cumsum(delta, dim=0)             # (C,)
        # mu[0]=0 固定、mu[c] = cumsum[c-1]/total * 2π
        mu = torch.cat([
            torch.zeros(1, device=delta.device),
            2.0 * np.pi * cumsum[:-1] / total
        ])
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