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