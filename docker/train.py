import os
import random
import argparse
import time
from pathlib import Path
import multiprocessing as mp

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

# ========= Hilos / Núcleos =========
def set_threads(n: int):
    n = max(1, int(n))
    # PyTorch
    torch.set_num_threads(n)                      # intra-op
    torch.set_num_interop_threads(max(1, n // 2)) # inter-op (más estable)
    # BLAS / OpenMP / Accelerate
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(n)   # macOS Accelerate
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)

# ====== Modelo con optimizaciones ======
class SmallCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, groups=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1, groups=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 32x16x16
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 64x8x8
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 128x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ====== Dataset con optimizaciones ======
def get_dataloaders(data_root, batch_size, num_workers=None, pin_memory=False):
    # Si no pasan workers, igualar a cores del sistema (se sobreescribe desde main)
    if num_workers is None:
        num_workers = max(1, os.cpu_count() or 4)

    # Params seguros según num_workers
    pw_train = bool(num_workers > 0)
    pf_train = 4 if num_workers > 0 else None

    num_workers_val = max(1, num_workers // 2)
    pw_val = bool(num_workers_val > 0)
    pf_val = 2 if num_workers_val > 0 else None

    normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        normalize
    ])
    test_tf = T.Compose([T.ToTensor(), normalize])

    train_ds = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_ds = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)

    # Train loader
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=pw_train,
        drop_last=True,
        **({"prefetch_factor": pf_train} if pf_train is not None else {})
    )

    # Val loader (menos presión)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers_val,
        pin_memory=pin_memory,
        persistent_workers=pw_val,
        **({"prefetch_factor": pf_val} if pf_val is not None else {})
    )
    return train_loader, test_loader

# ====== Utils ======
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def pick_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "auto":
        if torch.backends.mps.is_available():
            console.print("[yellow]MPS disponible, pero usando CPU por petición (núcleos fijos)[/yellow]")
            return torch.device("cpu")
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

# ====== Compilación opcional ======
def compile_model(model, device):
    try:
        import sys
        python_version = sys.version_info
        if python_version >= (3, 13) and sys.platform == "darwin":
            console.print("[yellow]torch.compile no estable en macOS con Python 3.13+; usando TorchScript[/yellow]")
            model = torch.jit.script(model)
            console.print("[green]Modelo compilado con TorchScript[/green]")
        elif hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead', dynamic=False)
            console.print("[green]Modelo compilado con torch.compile[/green]")
        else:
            model = torch.jit.script(model)
            console.print("[green]Modelo compilado con TorchScript[/green]")
    except Exception as e:
        console.print(f"[yellow]No se pudo compilar el modelo: {e}[/yellow]")
        console.print("[yellow]Continuando sin compilación...[/yellow]")
    return model

# ====== Entrenamiento ======
def train(args):
    # Fijar EXACTAMENTE los núcleos solicitados
    set_threads(args.cores)
    set_seed(args.seed)

    device = pick_device(args.device if not args.cpu else "cpu")
    console.print(f"Using device: [bold]{device}[/bold]")
    console.print(f"Núcleos solicitados: [bold]{args.cores}[/bold]")
    console.print(f"PyTorch intra-op threads: [bold]{torch.get_num_threads()}[/bold]")

    # En CPU: pin_memory=False
    pin = False

    # Si no pasaron --workers, igualar a --cores
    if args.workers is None:
        args.workers = args.cores

    data_root = os.path.expanduser(args.data)
    train_loader, test_loader = get_dataloaders(
        data_root,
        args.batch_size,
        num_workers=args.workers,
        pin_memory=pin
    )

    model = SmallCIFAR().to(device)

    if args.compile and not args.no_compile:
        model = compile_model(model, device)
    elif args.no_compile:
        console.print("[yellow]Compilación desactivada por --no-compile[/yellow]")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )

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

            for batch_idx, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                running_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / total
            train_acc = correct / total
            val_loss, val_acc = evaluate(model, test_loader, device)

            epoch_time = time.time() - epoch_start
            progress.advance(task)
            console.print(
                f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss:.4f} acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} acc={val_acc:.4f} | lr={scheduler.get_last_lr()[0]:.6f} | time={epoch_time:.2f}s"
            )

            if val_acc > best_acc:
                best_acc = val_acc
                ckpt_path = ckpt_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "val_acc": val_acc,
                }, ckpt_path)

    total_time = time.time() - total_start
    avg_epoch_time = total_time / args.epochs
    samples_per_sec = len(train_loader.dataset) * args.epochs / total_time

    table = Table(title="Resultados finales")
    table.add_column("Best Val Acc", justify="right")
    table.add_column("Total Time (s)", justify="right")
    table.add_column("Avg Epoch Time (s)", justify="right")
    table.add_column("Samples/sec", justify="right")
    table.add_row(f"{best_acc:.4f}", f"{total_time:.2f}", f"{avg_epoch_time:.2f}", f"{samples_per_sec:.0f}")
    console.print(table)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)  # recomendado en macOS

    parser = argparse.ArgumentParser(description="CIFAR-10 optimizado para Mac M4 (núcleos fijos)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--workers", type=int, default=None, help="Data loader workers (None=usa --cores)")
    parser.add_argument("--cpu", action="store_true", help="Forzar CPU")
    parser.add_argument("--compile", action="store_true", help="Compilar modelo (TorchScript/compile si aplica)")
    parser.add_argument("--no-compile", action="store_true", help="Desactivar compilación completamente")
    parser.add_argument("--data", type=str, default="~/.torch-datasets", help="Ruta cache CIFAR-10")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["auto","cpu","mps"], help="Selecciona dispositivo")
    parser.add_argument("--cores", type=int, default=4, help="🔥 Núcleos/hilos de CPU a usar")

    args = parser.parse_args()

    console.print("[bold]Configuración de ejecución:[/bold]")
    console.print(f"• Cores: {args.cores}")
    console.print(f"• Batch size: {args.batch_size}")
    console.print(f"• Data workers: {args.workers or f'auto→{args.cores}'}")
    console.print(f"• Compilación: {'ON' if args.compile and not args.no_compile else 'OFF' if args.no_compile else 'ON (auto)'}")

    train(args)
