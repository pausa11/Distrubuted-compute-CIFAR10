import os
import random
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from rich.progress import Progress
from rich.table import Table
from rich.console import Console
import torch.nn.functional as F

console = Console()

# ====== Modelo ======
class SmallCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 128x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ====== Dataset ======
def get_dataloaders(data_root, batch_size, num_workers=2, pin_memory=False):
    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), normalize])
    test_tf  = T.Compose([T.ToTensor(), normalize])

    train_ds = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, test_loader

# ====== Utils ======
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def pick_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device("cpu")

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total

# ====== Entrenamiento ======
def train(args):
    set_seed(args.seed)

    device = pick_device(args.device if not args.cpu else "cpu")
    console.print(f"Using device: [bold]{device}[/bold]")

    pin = False
    data_root = os.path.expanduser(args.data)
    train_loader, test_loader = get_dataloaders(data_root, args.batch_size, args.workers, pin_memory=pin)

    model = SmallCIFAR().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    ckpt_dir = Path(args.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    with Progress() as progress:
        task = progress.add_task("Training", total=args.epochs)
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()

            model.train()
            running_loss = 0.0
            total = 0
            correct = 0

            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc  = correct / total
            val_loss, val_acc = evaluate(model, test_loader, device)
            scheduler.step()

            epoch_time = time.time() - epoch_start
            progress.advance(task)
            console.print(
                f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} acc={val_acc:.4f} | time={epoch_time:.2f}s"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = ckpt_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, ckpt_path)

    total_time = time.time() - total_start
    table = Table(title="Resultados finales")
    table.add_column("Best Val Acc", justify="right")
    table.add_column("Total Time (s)", justify="right")
    table.add_row(f"{best_acc:.4f}", f"{total_time:.2f}")
    console.print(table)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 local training")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU")
    parser.add_argument("--data", type=str, default="~/.torch-datasets", help="Ruta cache CIFAR-10")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","mps"], help="Selecciona dispositivo")
    args = parser.parse_args()

    # Ejecuta el flujo completo desde aquí
    train(args)
