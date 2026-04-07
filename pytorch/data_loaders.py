import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from pathlib import Path
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from typing import List, Dict

PHENOCAM_DIR = Path('/home/shunsuke/data/raw/phenocam/phenocamdata/ashburnham')
COIL_DIR = Path(__file__).parent.parent / 'data' / 'coil-100' / 'coil-100'
JURKAT_DIR = Path('/home/shunsuke/data/raw/extracted/jurkat_cell_cycle')
SYSMEX_DIR = Path(__file__).parent.parent / 'data' / 'sysmex_cell_cycle_3cls'
SYSMEX_7CLASS_DIR = Path(__file__).parent.parent / 'data' / 'dataset_preprocessed_7classes_mokushi_screening'

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
    def __init__(self, images, labels, transform=None, normalize=True):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if len(image.shape) == 2:  # グレースケール画像の場合
            image = torch.from_numpy(image).unsqueeze(0).float()  # チャンネル次元を追加
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float()  # (H, W, C) -> (C, H, W)

        if self.normalize:
            image = image / 255.0  # 0-255の範囲を0-1に正規化

        if self.transform:
            image = self.transform(image)

        return image, label

def get_cfv_loader(batch_size=64, num_workers=2, num_classes=8, image_size=224):
    from datasets import load_dataset

    def load_split(split):
        X, y = [], []
        for sample in ds[split]:
            img = sample['image']
            angle = sample['angle']

            offset = 360 / (2 * num_classes)
            label = int((angle % 360 + offset) // (360 / num_classes)) % num_classes

            img = img.resize((image_size, image_size))
            img_array = np.array(img)

            X.append(img_array)
            y.append(label)

        return np.array(X), np.array(y)

    cache_path = Path(__file__).parent.parent / 'data' / 'cfv_cache.npz'
    if cache_path.exists():
        data = np.load(cache_path)
        X_train, y_train = data['X_train'], data['y_train']
        X_test, y_test = data['X_test'], data['y_test']
    else:
        ds = load_dataset("fort-cyber/CFV-Dataset")
        X_train, y_train = load_split('train')
        X_test, y_test = load_split('test')
        np.savez(cache_path, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    train_dataset = ImageDataset(X_train, y_train)
    val_dataset = ImageDataset(X_val, y_val)
    test_dataset = ImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def get_coil_loader(batch_size=64, num_workers=2, num_classes=8, image_size=128):
    from PIL import Image
    X, y = [], []
    for img in sorted(COIL_DIR.glob('obj*__*.png')):
        label = int(img.stem.split('__')[1]) // (360 // num_classes)
        img = Image.open(img).resize((image_size, image_size))
        img = np.array(img)
        X.append(img)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    train_dataset = ImageDataset(X_train, y_train)
    val_dataset = ImageDataset(X_val, y_val)
    test_dataset = ImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)    

    return train_loader, val_loader, test_loader

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

def merge_4class(labels):
    """7クラス→4クラス: G1, S, G2, M(Pro+Meta+Ana+Telo)"""
    label_mapping = {
        0: 0,  # G1 -> G1
        1: 1,  # S -> S
        2: 2,  # G2 -> G2
        3: 3,  # Pro -> M
        4: 3,  # Meta -> M
        5: 3,  # Ana -> M
        6: 3   # Telo -> M
    }
    return np.array([label_mapping[label] for label in labels])

def get_jurkat_loaders(batch_size=64, limit_per_phase=None, num_workers=2, num_classes=3, image_size=66):

    PHASES3 = ['G1', 'S', 'G2/M']
    PHASES4 = ['G1', 'S', 'G2', 'M']
    PHASES7 = ['G1', 'S', 'G2', 'Prophase', 'Metaphase', 'Anaphase', 'Telophase']


    X: List[np.ndarray] = []
    phase_labels: List[str] = []
    phase_counts: Dict[str, int] = {p: 0 for p in PHASES7}

    # Brightfield Ch3 JPEG files
    patterns = ['*Ch3*.jpg', '*Ch3*.jpeg']

    for ph in PHASES7:
        files: List[Path] = []
        for pat in patterns:
            files.extend(sorted((JURKAT_DIR / ph).glob(pat)))

        # Deduplicate while preserving order
        seen = set()
        uniq_files = []
        for p in files:
            if p not in seen:
                uniq_files.append(p)
                seen.add(p)

        for p in uniq_files:
            if limit_per_phase is not None and phase_counts[ph] >= limit_per_phase:
                break
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape[0] != image_size or img.shape[1] != image_size:
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            X.append(img[..., None])
            phase_labels.append(ph)
            phase_counts[ph] += 1

    X = np.stack(X, axis=0)
    labels = np.array(phase_labels)

    label_to_index = {label: i for i, label in enumerate(PHASES7)}
    y = np.array([label_to_index[label] for label in labels])

    if num_classes == 3:
        y = merge_jurkat_3class(y)
        print(f"Merged labels into 3 classes: {PHASES3}")
    elif num_classes == 4:
        y = merge_4class(y)
        print(f"Merged labels into 4 classes: {PHASES4}")
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

def get_sysmex_7class_loaders(batch_size=64, num_workers=0, num_classes=7):
    # 3クラスのデータセットとは異なりtrainとtestに分かれていないため別関数として定義した
    PHASES7 = ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']
    PHASES4 = ['G1', 'S', 'G2', 'M']

    X, y = [], []
    label_to_index = {phase: idx for idx, phase in enumerate(PHASES7)}

    for phase in PHASES7:
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

    if num_classes == 4:
        y = merge_4class(y)
        print(f"Loaded {len(X)} images, merged into 4 classes: {PHASES4}")
    else:
        print(f"Loaded {len(X)} images from 7 phases: {PHASES7}")

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

def month_to_season(month: int) -> str:
    if month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Fall'
    else:
        return 'Winter'
    
def get_phenocam_loaders(batch_size=64, limit_per_season=None, image_size=224, num_workers=2, label_type='season'):
    SEASONS = ['Spring', 'Summer', 'Fall', 'Winter']
    MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    X: List[np.ndarray] = []
    labels_list: List[str] = []
    
    classes = SEASONS if label_type == 'season' else MONTHS
    class_counts: Dict[str, int] = {c: 0 for c in classes}

    for year_dir in sorted(PHENOCAM_DIR.glob('[0-9]*')):
        if not year_dir.is_dir():
            continue

        for month_dir in sorted(year_dir.glob('[0-9]*')):
            if not month_dir.is_dir():
                continue

            month = int(month_dir.name)
            
            if label_type == 'season':
                label = month_to_season(month)
            elif label_type == 'month':
                # month is 1-12, MONTHS is 0-indexed
                label = MONTHS[month - 1]
            else:
                raise ValueError(f"Unknown label_type: {label_type}")

            if limit_per_season is not None and class_counts[label] >= limit_per_season:
                continue

            for img_path in sorted(month_dir.glob('*.jpg')):
                # クラス制限の再チェック（月フォルダ内に複数画像があるため）
                if limit_per_season is not None and class_counts[label] >= limit_per_season:
                    break

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                if img.shape[0] != image_size or img.shape[1] != image_size:
                    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)

                # 学習しやすいように[0, 255]から[0, 1]に正規化
                img = img.astype(np.float32) / 255.0

                X.append(img)
                labels_list.append(label)
                class_counts[label] += 1

    X = np.stack(X, axis=0)
    labels = np.array(labels_list)

    label_to_index = {label: i for i, label in enumerate(classes)}
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

    train_dataset = ImageDataset(X_train, y_train, normalize=False)
    val_dataset = ImageDataset(X_val, y_val, normalize=False)
    test_dataset = ImageDataset(X_test, y_test, normalize=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader