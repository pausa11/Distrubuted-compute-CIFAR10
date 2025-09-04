# main.py (entrenamiento distribuido DDP para CIFAR-10)
import os, json, time, argparse, datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as T

# ========= Setup DDP =========
def setup_ddp():
    # Lee las variables que enviará el agent
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # En Windows no hay NCCL; usa gloo. En Linux con GPU usa nccl
    backend = "gloo"
    if os.name != "nt" and torch.cuda.is_available():
        backend = "nccl"

    # Dispositivo
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and os.name != "nt" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    # Fuerza NO libuv en el TCPStore de rendezvous (clave para Windows)
    init_method = f"tcp://{master_addr}:{master_port}"
    
    # Configurar timeout más largo para conexiones lentas
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=1800),
    )

    return rank, world_size, local_rank, device

# ========= Modelo =========
class SmallResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.m = torchvision.models.resnet18(num_classes=num_classes)
        # Ajustes para CIFAR-10 (imágenes 32x32)
        self.m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.m.maxpool = nn.Identity()

    def forward(self, x):
        return self.m(x)

# ========= Dataloaders =========
def get_dataloaders(world_size, rank, data_root="./data", batch_per_proc=128, num_workers=2):
    normalize = T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    # Solo descargar en rank 0
    download = (rank == 0)
    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=download, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=download, transform=test_tf)

    # Esperar a que rank 0 descargue
    if world_size > 1:
        dist.barrier()

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    # Reducir num_workers para evitar problemas en algunos sistemas
    dl_args = dict(batch_size=batch_per_proc, num_workers=num_workers, pin_memory=(torch.cuda.is_available()))
    train_loader = DataLoader(train_set, sampler=train_sampler, **dl_args)
    test_loader = DataLoader(test_set, sampler=test_sampler, **dl_args)
    return train_loader, test_loader, train_sampler

# ========= Evaluación =========
@torch.no_grad()
def evaluate(model, loader, device, world_size):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    
    # Reducir métricas entre procesos
    if world_size > 1:
        t = torch.tensor([correct, total], dtype=torch.long, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        correct, total = t[0].item(), t[1].item()
    
    return (correct / total) * 100.0 if total > 0 else 0.0

# ========= Guardar métricas =========
def write_metrics(epoch, loss, acc):
    path = os.environ.get("DDP_METRICS_FILE", "/tmp/ddp_metrics.json")
    if int(os.environ.get("RANK", "0")) == 0:  # solo rank 0 escribe
        metrics = {
            "epoch": int(epoch), 
            "loss": float(loss), 
            "acc": float(acc),
            "timestamp": time.time()
        }
        try:
            with open(path, "w") as f:
                json.dump(metrics, f)
        except Exception as e:
            print(f"Error writing metrics: {e}")

# ========= Args =========
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-per-proc", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--data-root", type=str, default="./data")
    return ap.parse_args()

# ========= Entrenamiento =========
def main():
    try:
        args = parse_args()
        rank, world_size, local_rank, device = setup_ddp()

        if rank == 0:
            print(f"Starting training on {world_size} processes")
            print(f"Device: {device}, Backend: {dist.get_backend()}")

        train_loader, test_loader, train_sampler = get_dataloaders(
            world_size, rank, data_root=args.data_root, batch_per_proc=args.batch_per_proc
        )

        model = SmallResNet18().to(device)
        
        # Solo usar DDP si hay múltiples procesos
        if world_size > 1:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None,
                       broadcast_buffers=False)

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        criterion = nn.CrossEntropyLoss()
        
        # Solo usar scaler si hay GPU disponible
        use_amp = device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_sampler.set_epoch(epoch)
            
            running_loss = 0.0
            num_samples = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                    loss = criterion(logits, y)
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * y.size(0)
                num_samples += y.size(0)
                
                # Log progreso cada 50 batches
                if rank == 0 and batch_idx % 50 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Calcular pérdida promedio
            if world_size > 1:
                loss_tensor = torch.tensor([running_loss, num_samples], dtype=torch.float64, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = (loss_tensor[0] / loss_tensor[1]).item()
            else:
                avg_loss = running_loss / num_samples

            # Evaluación
            acc = evaluate(model, test_loader, device, world_size)
            scheduler.step()

            if rank == 0:
                print(f"Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f} acc={acc:.2f}% lr={scheduler.get_last_lr()[0]:.6f}")
                write_metrics(epoch, avg_loss, acc)

        if rank == 0:
            print("Training completed successfully!")

    except Exception as e:
        print(f"Error in main: {e}")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()