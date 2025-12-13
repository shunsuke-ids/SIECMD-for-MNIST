import torch  
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import time
import wandb
from pathlib import Path

from losses import SoftmaxVectorLoss
from models import SimpleCNN
from datasets import get_mnist_loaders, get_jurkat_loaders

DATASETS = {
    'mnist': {'num_classes': 10, 'channels': 1, 'size': 28},
    'jurkat': {'num_classes': 3, 'channels': 1, 'size': 66}
}

LOSS_FUNCTIONS = {
    'ce': ('CrossEntropyLoss', nn.CrossEntropyLoss),
    'svl': ('SoftmaxVectorLoss', SoftmaxVectorLoss)
}

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # Pytorchは勾配の初期化が必要
        logits = model(inputs) # モデルの順伝播

        loss = loss_fn(logits, labels) # 損失の計算
        loss.backward() # 逆伝播(Pytorchはbackwardメソッド実装なしで呼び出せる)
        optimizer.step() # パラメータの更新

        total_loss += loss.item() * inputs.size(0) # バッチの損失を合計
        correct += (logits.argmax(dim=1) == labels).sum().item() # 正解数を合計
        total += inputs.size(0) # サンプル数を合計
    
    return total_loss / total, correct / total # 平均損失と精度を返す

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)

            loss = loss_fn(logits, labels)

            total_loss += loss.item() * inputs.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += inputs.size(0)
    
    return total_loss / total, correct / total

def train_and_evaluate(model, train_loader, test_loader, loss_fn,
                     optimizer, device, epochs, loss_name):
    print(f"\n{'='*60}")
    print(f"Training with {loss_name} for {epochs} epochs")
    print(f"{'='*60}\n")

    best_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }

    for epoch in range(epochs):
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        
        best_acc = max(best_acc, test_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)

        # エポックごとにwandbにログを記録
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        print(f"Epoch {epoch+1:2d}/{epochs} ({time.time()-start:.1f}s) | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Test: {test_loss:.4f}/{test_acc:.4f} | Best: {best_acc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Best Test Accuracy: {best_acc:.4f}")
    print(f"{'='*60}\n")

    # 最終結果をwandbのsummaryに記録
    wandb.summary["best_test_acc"] = best_acc

    return history, best_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), required=True)
    parser.add_argument('--loss', type=str, choices=LOSS_FUNCTIONS.keys(), required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--limit_per_phase', type=int, default=None)

    args = parser.parse_args()

    set_seed()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    cfg = DATASETS[args.dataset]
    val_loader = None
    if args.dataset == 'mnist':
        train_loader, test_loader = get_mnist_loaders(args.batch_size)
    elif args.dataset == 'jurkat':
        train_loader, val_loader,test_loader = get_jurkat_loaders(args.batch_size, limit_per_phase=args.limit_per_phase, num_classes=cfg['num_classes'])
    print(f"Dataset: {args.dataset.upper()} | Train: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset) if val_loader is not None else 0} |Test: {len(test_loader.dataset)}")

    wandb.init(
        project="ce_vs_svl",
        group=args.dataset,
        tags=[args.loss],
        name=f"{args.loss}_{args.lr}lr_{args.epochs}ep",
        config=vars(args)
    )

    model = SimpleCNN(cfg['channels'], cfg['num_classes'], cfg['size']).to(device)
    
    loss_name, loss_fn_class = LOSS_FUNCTIONS[args.loss]
    if args.loss == 'svl':
        loss_fn = loss_fn_class(num_classes=cfg['num_classes']).to(device)
    else:
        loss_fn = loss_fn_class()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    eval_loader = val_loader if val_loader is not None else test_loader

    train_and_evaluate(
        model, train_loader, eval_loader, loss_fn,
        optimizer, device, args.epochs, loss_name
    )

if __name__ == '__main__':
    main()

