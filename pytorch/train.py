import torch  
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import time
import wandb
from pathlib import Path
import numpy as np
import random
import copy
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from losses import EuclideanVectorLoss, NormalizedSoftmaxVectorLoss, SoftmaxVectorLoss, MSEVectorLoss
from models import SimpleCNN
from datasets import get_mnist_loaders, get_jurkat_loaders, get_sysmex_loaders, get_sysmex_7class_loaders, get_phenocam_loaders
from metrics import circular_mae, circular_mae_per_class

DATASETS = {
    'mnist': {'num_classes': 10, 'channels': 1, 'size': 28, 'class_names': [str(i) for i in range(10)]},
    'jurkat': {'num_classes': 3, 'channels': 1, 'size': 66, 'class_names': ['G1', 'S', 'G2/M']},
    'jurkat4': {'num_classes': 4, 'channels': 1, 'size': 66, 'class_names': ['G1', 'S', 'G2', 'M']},
    'jurkat7': {'num_classes': 7, 'channels': 1, 'size': 66, 'class_names': ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']},
    'sysmex': {'num_classes': 3, 'channels': 3, 'size': 64, 'class_names': ['G1', 'S', 'G2']},
    'sysmex4': {'num_classes': 4, 'channels': 3, 'size': 64, 'class_names': ['G1', 'S', 'G2', 'M']},
    'sysmex7': {'num_classes': 7, 'channels': 3, 'size': 64, 'class_names': ['G1', 'S', 'G2', 'Pro', 'Meta', 'Ana', 'Telo']},
    'phenocam': {'num_classes': 4, 'channels': 3, 'size': 224, 'class_names': ['Spring', 'Summer', 'Fall', 'Winter']},
    'phenocam_monthly': {'num_classes': 12, 'channels': 3, 'size': 224, 'class_names': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']}
}

LOSS_FUNCTIONS = {
    'ce': ('CrossEntropyLoss', nn.CrossEntropyLoss),
    'svl': ('SoftmaxVectorLoss', SoftmaxVectorLoss),
    'nsvl': ('NormalizedSoftmaxVectorLoss', NormalizedSoftmaxVectorLoss),
    'msevl': ('MSEVectorLoss', MSEVectorLoss),
    'eucvl': ('EuclideanVectorLoss', EuclideanVectorLoss)
}

def set_seed(seed=42):
    """完全な再現性のためのシード設定"""
    # Python標準の乱数
    random.seed(seed)

    # Numpyの乱数
    np.random.seed(seed)

    # PyTorchの乱数
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # CuDNNの決定的動作を有効化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # PyTorch 1.8以降: 決定的アルゴリズムを使用
    # 注意: 一部の操作で性能が低下する可能性があります
    # torch.use_deterministic_algorithms(True)  # 必要に応じてコメント解除

def seed_worker(_worker_id):
    """DataLoaderのworkerごとにシードを設定"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

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

def evaluate_detailed(model, loader, device, num_classes,  class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # メトリクスの計算
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)

    # F1スコア
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # MAE
    circ_mae = circular_mae(all_preds, all_labels, num_classes)
    circ_mae_per_class = circular_mae_per_class(all_preds, all_labels, num_classes)

    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'circular_mae': circ_mae,
        'circular_mae_per_class': circ_mae_per_class['per_class'],
        'circular_mae_macro': circ_mae_per_class['macro'],
        'predictions': all_preds,
        'labels': all_labels
    }

def train_and_evaluate(model, train_loader, val_loader, test_loader, loss_fn,
                     optimizer, device, epochs, loss_name, num_classes, class_names=None,
                     dataset_name=None, loss_key=None, save_dir='./saved_models', patience=None):
    print(f"\n{'='*60}")
    print(f"Training with {loss_name} for {epochs} epochs")
    if patience:
        print(f"Early Stopping: patience={patience}")
    print(f"{'='*60}\n")

    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    early_stop_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(epochs):
        start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device)

        # ベストモデルの保存（Accuracyで判定）
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        # Early Stopping（Lossで判定）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # エポックごとにwandbにログを記録
        wandb.log({
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        print(f"Epoch {epoch+1:2d}/{epochs} ({time.time()-start:.1f}s) | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | Best Acc: {best_val_acc:.4f} | Best Loss: {best_val_loss:.4f}")

        # Early Stopping判定
        if patience and early_stop_counter >= patience:
            print(f"\nEarly Stopping at epoch {epoch+1} (no improvement in val_loss for {patience} epochs)")
            break

    actual_epochs = len(history['train_loss'])
    print(f"\n{'='*60}")
    print(f"Training finished at epoch {actual_epochs}/{epochs}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    # 最終エポックのモデルを保存
    final_model_state = copy.deepcopy(model.state_dict())

    # ===========================================
    # 1. ベストモデル（Validation最高）でTest set評価
    # ===========================================
    print("=" * 60)
    print("【Best Model (Best Validation Acc)】")
    print("=" * 60)
    model.load_state_dict(best_model_state)

    best_test_loss, best_test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {best_test_loss:.4f} | Test Accuracy: {best_test_acc:.4f}")

    best_detailed_metrics = evaluate_detailed(model, test_loader, device, num_classes, class_names)

    print("\n混同行列 (Test set - Best Model):")
    print(best_detailed_metrics['confusion_matrix'])

    print("\nクラスごとの詳細メトリクス (Test set - Best Model):")
    best_report = best_detailed_metrics['classification_report']
    if class_names:
        for class_name in class_names:
            if class_name in best_report:
                metrics = best_report[class_name]
                print(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                      f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    print(f"\nマクロ平均 F1: {best_detailed_metrics['f1_macro']:.4f}")
    print(f"加重平均 F1: {best_detailed_metrics['f1_weighted']:.4f}")
    print(f"Accuracy: {best_report['accuracy']:.4f}")
    print(f"Circular MAE: {best_detailed_metrics['circular_mae']:.4f}")
    print(f"Circular MAE (Macro): {best_detailed_metrics['circular_mae_macro']:.4f}")

    # クラスごとのcMAE
    print("\nクラスごとの Circular MAE:")
    if class_names:
        for i, class_name in enumerate(class_names):
            cmae_val = best_detailed_metrics['circular_mae_per_class'].get(i, float('nan'))
            print(f"  {class_name}: {cmae_val:.4f}")

    # ベストモデルの混同行列を画像として保存
    fig_best, ax_best = plt.subplots(figsize=(10, 8))
    sns.heatmap(best_detailed_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(best_detailed_metrics['confusion_matrix'])),
                yticklabels=class_names if class_names else range(len(best_detailed_metrics['confusion_matrix'])),
                ax=ax_best)
    ax_best.set_xlabel('Predicted')
    ax_best.set_ylabel('True')
    ax_best.set_title('Confusion Matrix (Test set - Best Model)')
    wandb.log({"confusion_matrix_best": wandb.Image(fig_best)})
    plt.close(fig_best)

    # ===========================================
    # 2. 最終エポックのモデルでTest set評価
    # ===========================================
    print("\n" + "=" * 60)
    print("【Final Model (Last Epoch)】")
    print("=" * 60)
    model.load_state_dict(final_model_state)

    final_test_loss, final_test_acc = evaluate(model, test_loader, loss_fn, device)
    print(f"Test Loss: {final_test_loss:.4f} | Test Accuracy: {final_test_acc:.4f}")

    final_detailed_metrics = evaluate_detailed(model, test_loader, device, num_classes, class_names)

    print("\n混同行列 (Test set - Final Model):")
    print(final_detailed_metrics['confusion_matrix'])

    print("\nクラスごとの詳細メトリクス (Test set - Final Model):")
    final_report = final_detailed_metrics['classification_report']
    if class_names:
        for class_name in class_names:
            if class_name in final_report:
                metrics = final_report[class_name]
                print(f"  {class_name}: Precision={metrics['precision']:.4f}, "
                      f"Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")

    print(f"\nマクロ平均 F1: {final_detailed_metrics['f1_macro']:.4f}")
    print(f"加重平均 F1: {final_detailed_metrics['f1_weighted']:.4f}")
    print(f"Accuracy: {final_report['accuracy']:.4f}")
    print(f"Circular MAE: {final_detailed_metrics['circular_mae']:.4f}")
    print(f"Circular MAE (Macro): {final_detailed_metrics['circular_mae_macro']:.4f}")

    # クラスごとのcMAE
    print("\nクラスごとの Circular MAE:")
    if class_names:
        for i, class_name in enumerate(class_names):
            cmae_val = final_detailed_metrics['circular_mae_per_class'].get(i, float('nan'))
            print(f"  {class_name}: {cmae_val:.4f}")

    # 最終エポックモデルの混同行列を画像として保存
    fig_final, ax_final = plt.subplots(figsize=(10, 8))
    sns.heatmap(final_detailed_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else range(len(final_detailed_metrics['confusion_matrix'])),
                yticklabels=class_names if class_names else range(len(final_detailed_metrics['confusion_matrix'])),
                ax=ax_final)
    ax_final.set_xlabel('Predicted')
    ax_final.set_ylabel('True')
    ax_final.set_title('Confusion Matrix (Test set - Final Model)')
    wandb.log({"confusion_matrix_final": wandb.Image(fig_final)})
    plt.close(fig_final)

    # ===========================================
    # wandbのsummaryに記録
    # ===========================================
    # 学習情報
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["actual_epochs"] = actual_epochs
    wandb.summary["early_stopped"] = actual_epochs < epochs

    # 混同行列の数値データ
    wandb.summary["best_confusion_matrix"] = best_detailed_metrics['confusion_matrix'].tolist()
    wandb.summary["final_confusion_matrix"] = final_detailed_metrics['confusion_matrix'].tolist()

    # ベストモデルの結果
    wandb.summary["best_val_acc"] = best_val_acc
    wandb.summary["best_test_acc"] = best_test_acc
    wandb.summary["best_test_loss"] = best_test_loss
    wandb.summary["best_f1_macro"] = best_detailed_metrics['f1_macro']
    wandb.summary["best_f1_weighted"] = best_detailed_metrics['f1_weighted']
    wandb.summary["best_circular_mae"] = best_detailed_metrics['circular_mae']
    wandb.summary["best_circular_mae_macro"] = best_detailed_metrics['circular_mae_macro']

    # 最終エポックモデルの結果
    wandb.summary["final_val_acc"] = history['val_acc'][-1]
    wandb.summary["final_test_acc"] = final_test_acc
    wandb.summary["final_test_loss"] = final_test_loss
    wandb.summary["final_f1_macro"] = final_detailed_metrics['f1_macro']
    wandb.summary["final_f1_weighted"] = final_detailed_metrics['f1_weighted']
    wandb.summary["final_circular_mae"] = final_detailed_metrics['circular_mae']
    wandb.summary["final_circular_mae_macro"] = final_detailed_metrics['circular_mae_macro']

    # クラスごとのメトリクス
    if class_names:
        for i, class_name in enumerate(class_names):
            # ベストモデル
            if class_name in best_report:
                wandb.summary[f"best_{class_name}_f1"] = best_report[class_name]['f1-score']
                wandb.summary[f"best_{class_name}_precision"] = best_report[class_name]['precision']
                wandb.summary[f"best_{class_name}_recall"] = best_report[class_name]['recall']
            if i in best_detailed_metrics['circular_mae_per_class']:
                wandb.summary[f"best_{class_name}_cmae"] = best_detailed_metrics['circular_mae_per_class'][i]

            # 最終エポックモデル
            if class_name in final_report:
                wandb.summary[f"final_{class_name}_f1"] = final_report[class_name]['f1-score']
                wandb.summary[f"final_{class_name}_precision"] = final_report[class_name]['precision']
                wandb.summary[f"final_{class_name}_recall"] = final_report[class_name]['recall']
            if i in final_detailed_metrics['circular_mae_per_class']:
                wandb.summary[f"final_{class_name}_cmae"] = final_detailed_metrics['circular_mae_per_class'][i]

    # ===========================================
    # モデルの保存
    # ===========================================
    if dataset_name and loss_key:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # ベストモデルを保存
        best_model_path = save_path / f"{dataset_name}_{loss_key}_{epochs}ep_best.pth"
        torch.save(best_model_state, best_model_path)
        print(f"\nBest model saved: {best_model_path}")

        # 最終エポックモデルを保存
        final_model_path = save_path / f"{dataset_name}_{loss_key}_{epochs}ep_final.pth"
        torch.save(final_model_state, final_model_path)
        print(f"Final model saved: {final_model_path}")

        # Wandbにモデルパスを記録
        wandb.summary["best_model_path"] = str(best_model_path)
        wandb.summary["final_model_path"] = str(final_model_path)

    return history, best_val_acc, best_test_acc, final_test_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=DATASETS.keys(), required=True)
    parser.add_argument('--loss', type=str, choices=LOSS_FUNCTIONS.keys(), required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience (default: None = no early stopping)')
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
        train_loader, val_loader, test_loader = get_jurkat_loaders(args.batch_size, limit_per_phase=args.limit_per_phase, num_classes=3)
    elif args.dataset == 'jurkat4':
        train_loader, val_loader, test_loader = get_jurkat_loaders(args.batch_size, limit_per_phase=args.limit_per_phase, num_classes=4)
    elif args.dataset == 'jurkat7':
        train_loader, val_loader, test_loader = get_jurkat_loaders(args.batch_size, limit_per_phase=args.limit_per_phase, num_classes=7)
    elif args.dataset == 'sysmex':
        train_loader, test_loader = get_sysmex_loaders(args.batch_size)
    elif args.dataset == 'sysmex4':
        train_loader, val_loader, test_loader = get_sysmex_7class_loaders(args.batch_size, num_classes=4)
    elif args.dataset == 'sysmex7':
        train_loader, val_loader, test_loader = get_sysmex_7class_loaders(args.batch_size, num_classes=7)
    elif args.dataset == 'phenocam':
        train_loader, val_loader, test_loader = get_phenocam_loaders(args.batch_size, limit_per_season=args.limit_per_phase)
    elif args.dataset == 'phenocam_monthly':
        train_loader, val_loader, test_loader = get_phenocam_loaders(args.batch_size, limit_per_season=args.limit_per_phase, label_type='month')

    # val_loaderがない場合はtest_loaderを使う（mnist, sysmex）
    if val_loader is None:
        val_loader = test_loader
        print(f"Dataset: {args.dataset.upper()} | Train: {len(train_loader.dataset)} | Val/Test: {len(test_loader.dataset)} (no separate validation set)")
    else:
        print(f"Dataset: {args.dataset.upper()} | Train: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

    wandb.init(
        project="ce_vs_svl",
        group=args.dataset,
        tags=[args.loss],
        name=f"{args.loss}_{args.lr}lr_{args.epochs}ep",
        config=vars(args)
    )

    model = SimpleCNN(cfg['channels'], cfg['num_classes'], cfg['size']).to(device)

    loss_name, loss_fn_class = LOSS_FUNCTIONS[args.loss]
    if args.loss in ['svl', 'nsvl', 'msevl', 'eucvl']:
        loss_fn = loss_fn_class(num_classes=cfg['num_classes']).to(device)
    else:
        loss_fn = loss_fn_class()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_and_evaluate(
        model, train_loader, val_loader, test_loader, loss_fn,
        optimizer, device, args.epochs, loss_name, cfg['num_classes'], cfg['class_names'],
        dataset_name=args.dataset, loss_key=args.loss, patience=args.patience
    )

if __name__ == '__main__':
    main()

