import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

def get_mnist_loaders(batch_size=64, data_dir='./data', num_workers=2):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # この値で正規化すると平均が0, 分散が1になる
    ])

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader

class ImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if len(image.shape) == 2:  # グレースケール画像の場合
            image = torch.from_numpy(image).unsqueeze(0).float()  # チャンネル次元を追加
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)

        image = image / 255.0  # 0-255の範囲を0-1に正規化

        if self.transform:
            image = self.transform(image)

        return image, label

def merge_jurkat_3class(labels):
    label_mapping = {
        0:0,
        1:1,
        2:2,
        3:2,
        4:2,
        5:2,
        6:2
    }
    return np.array([label_mapping[label] for label in labels])

def get_jurkat_loaders(batch_size=64, limit_per_phase=None, num_workers=2, num_classes=3):
    from src.regression.utils.data_loaders import load_jurkat_ch3_data
    from sklearn.model_selection import train_test_split

    PHASES3 = ['G1', 'S', 'G2/M']

    X, labels = load_jurkat_ch3_data(limit_per_phase=limit_per_phase, image_size=66)

    from src.regression.utils.data_loaders import get_label_to_index_mapping, PHASES7
    label_to_index = get_label_to_index_mapping(PHASES7)
    y = np.array([label_to_index[label] for label in labels])

    if num_classes == 3:
        y = merge_jurkat_3class(y)
        print(f"Merged labels into 3 classes: {PHASES3}")
    else:
        print(f"Using original 7 classes: {PHASES7}")
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_dataset = ImageDataset(X_train, y_train)
    val_dataset = ImageDataset(X_val, y_val)
    test_dataset = ImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_sysmex_loaders(batch_size=64, num_workers=2):
    import cv2
    from pathlib import Path

    SYSMEX_DIR = Path(__file__).parent.parent / 'data' / 'sysmex_cell_cycle_3cls'
    PHASES = ['G1', 'S', 'G2']

    def load_split(split):
        X, y = [], []
        label_to_index = {phase: idx for idx, phase in enumerate(PHASES)}

        for phase in PHASES:
            phase_dir = SYSMEX_DIR / split / phase
            for tif_path in sorted(phase_dir.glob('*_merged.tif')):
                img = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                X.append(img)
                y.append(label_to_index[phase])

        return np.array(X), np.array(y)

    X_train, y_train = load_split('train')
    X_test, y_test = load_split('test')

    train_dataset = ImageDataset(X_train, y_train)
    test_dataset = ImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

def get_sysmex_7class_loaders(batch_size=64, num_workers=0):
    # 3クラスのデータセットとは異なりtrainとtestに分かれていないため別関数として定義した
    import cv2
    from pathlib import Path
    from sklearn.model_selection import train_test_split

    SYSMEX_7CLASS_DIR = Path(__file__).parent.parent / 'data' / 'dataset_preprocessed_7classes_mokushi_screening'
    PHASES = ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']

    X, y = [], []
    label_to_index = {phase: idx for idx, phase in enumerate(PHASES)}

    for phase in PHASES:
        phase_dir = SYSMEX_7CLASS_DIR / phase
        for tif_path in sorted(phase_dir.glob('*_merged.tif')):
            img = cv2.imread(str(tif_path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img)
            y.append(label_to_index[phase])

    X = np.array(X)
    y = np.array(y)

    print(f"Loaded {len(X)} images from 7 phases: {PHASES}")

    # 70:15:15の比率でtrain:val:testに分割
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_dataset = ImageDataset(X_train, y_train)
    val_dataset = ImageDataset(X_val, y_val)
    test_dataset = ImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
class NormalizedImageDataset(Dataset):
    """既に[0,1]に正規化済みの画像用Dataset"""
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if len(image.shape) == 2:  # グレースケール画像の場合
            image = torch.from_numpy(image).unsqueeze(0).float()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)

        # 既に[0,1]に正規化済みなので255で割らない

        if self.transform:
            image = self.transform(image)

        return image, label

def get_phenocam_loaders(batch_size=64, limit_per_season=None, image_size=224, num_workers=2, label_type='season'):
    from src.regression.utils.data_loaders import load_phenocam_data, get_label_to_index_mapping, SEASONS, MONTHS
    from sklearn.model_selection import train_test_split

    X, labels = load_phenocam_data(label_type=label_type, limit_per_class=limit_per_season, image_size=image_size)

    classes = SEASONS if label_type == 'season' else MONTHS
    label_to_index = get_label_to_index_mapping(classes)
    y = np.array([label_to_index[label] for label in labels])

    print(f"Loaded {len(X)} images from {len(classes)} {label_type}s")

    # クラス分布を表示
    unique, counts = np.unique(y, return_counts=True)
    for idx, count in zip(unique, counts):
        print(f"  {classes[idx]}: {count} images ({count/len(y)*100:.1f}%)")

    # 70:15:15の比率でtrain:val:testに分割
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    train_dataset = NormalizedImageDataset(X_train, y_train)
    val_dataset = NormalizedImageDataset(X_val, y_val)
    test_dataset = NormalizedImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
