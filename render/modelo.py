import os
import random
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F
from flask import Flask, request, jsonify
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

batch_manager = None
model = None

def ensure_initialized():
    global batch_manager, model
    if model is None:
        model = SmallCIFAR().to(torch.device("cpu"))
    if batch_manager is None:
        # Importar torchvision solo cuando haga falta (menos RAM en frío)
        import torchvision
        import torchvision.transforms as T
        data_root = os.environ.get('DATA_ROOT', '/tmp/torch-datasets')
        batch_manager = BatchManager(data_root)


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

# ====== Dataset y Batch Management ======
class BatchManager:
    def __init__(self, data_root: str, batch_size: int = 128):
        self.data_root = data_root
        self.batch_size = batch_size
        self.device = torch.device("cpu")  # Forzar CPU para Render
        
        # Cargar datasets
        self._load_datasets()
        
    def _load_datasets(self):
        """Carga los datasets CIFAR-10"""
        normalize = T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_tf = T.Compose([
            T.RandomCrop(32, padding=4), 
            T.RandomHorizontalFlip(), 
            T.ToTensor(), 
            normalize
        ])
        test_tf = T.Compose([T.ToTensor(), normalize])

        self.train_ds = torchvision.datasets.CIFAR10(
            root=self.data_root, train=True, download=True, transform=train_tf
        )
        self.test_ds = torchvision.datasets.CIFAR10(
            root=self.data_root, train=False, download=True, transform=test_tf
        )
        
        logger.info(f"Dataset cargado: {len(self.train_ds)} muestras de entrenamiento")
    
    def get_batch_loader(self, batch_indices: list) -> DataLoader:
        """Crea un DataLoader para los índices específicos del batch"""
        subset = Subset(self.train_ds, batch_indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def get_test_loader(self) -> DataLoader:
        """Retorna el DataLoader completo de test"""
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
    def get_batch_indices(self, batch_id: int, total_batches: int) -> list:
        """Calcula los índices para un batch específico"""
        total_samples = len(self.train_ds)
        samples_per_batch = total_samples // total_batches
        
        start_idx = batch_id * samples_per_batch
        end_idx = (batch_id + 1) * samples_per_batch if batch_id < total_batches - 1 else total_samples
        
        return list(range(start_idx, end_idx))

# ====== Training Utils ======
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate_batch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    ensure_initialized()
    """Evalúa el modelo en un batch específico"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

def train_batch(
    model: nn.Module, 
    batch_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    device: torch.device,
    model_state: Optional[Dict] = None
) -> Dict[str, Any]:
    """Entrena el modelo en un batch específico"""
    
    # Cargar estado del modelo si se proporciona
    if model_state:
        model.load_state_dict(model_state)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for images, labels in batch_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    training_time = time.time() - start_time
    
    return {
        "loss": total_loss / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
        "samples_processed": total,
        "training_time": training_time,
        "model_state": model.state_dict()
    }

# ====== Global Objects ======
batch_manager = None
model = None

def initialize_components():
    """Inicializa los componentes globales"""
    global batch_manager, model
    
    data_root = os.environ.get('DATA_ROOT', '/tmp/torch-datasets')
    batch_manager = BatchManager(data_root)
    model = SmallCIFAR().to(torch.device("cpu"))
    
    logger.info("Componentes inicializados correctamente")

# ====== API Endpoints ======
@app.route('/health', methods=['GET'])
def health_check():
    ensure_initialized()
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "Node is running"}), 200

@app.route('/info', methods=['GET'])
def get_info():
    ensure_initialized()
    """Información del nodo"""
    return jsonify({
        "device": "cpu",
        "model_params": sum(p.numel() for p in model.parameters()),
        "training_samples": len(batch_manager.train_ds),
        "test_samples": len(batch_manager.test_ds)
    }), 200

@app.route('/train_batch', methods=['POST'])
def train_batch_endpoint():
    ensure_initialized()
    """Endpoint principal para entrenar un batch específico"""
    try:
        data = request.get_json()
        
        # Validar parámetros requeridos
        required_params = ['batch_id', 'total_batches']
        for param in required_params:
            if param not in data:
                return jsonify({"error": f"Missing parameter: {param}"}), 400
        
        batch_id = data['batch_id']
        total_batches = data['total_batches']
        
        # Parámetros opcionales
        lr = data.get('lr', 0.1)
        momentum = data.get('momentum', 0.9)
        weight_decay = data.get('weight_decay', 5e-4)
        seed = data.get('seed', 1337)
        model_state = data.get('model_state')  # Estado del modelo para continuar entrenamiento
        
        # Validar rangos
        if batch_id < 0 or batch_id >= total_batches:
            return jsonify({"error": "batch_id out of range"}), 400
        
        # Configurar seed
        set_seed(seed)
        
        # Obtener índices del batch
        batch_indices = batch_manager.get_batch_indices(batch_id, total_batches)
        batch_loader = batch_manager.get_batch_loader(batch_indices)
        
        # Configurar optimizer
        optimizer = optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
        
        # Entrenar batch
        logger.info(f"Iniciando entrenamiento batch {batch_id}/{total_batches}")
        results = train_batch(
            model=model,
            batch_loader=batch_loader,
            optimizer=optimizer,
            device=torch.device("cpu"),
            model_state=model_state
        )
        
        # Convertir tensores a listas para serialización JSON
        model_state_serializable = {}
        for key, tensor in results["model_state"].items():
            model_state_serializable[key] = tensor.cpu().numpy().tolist()
        
        response = {
            "batch_id": batch_id,
            "total_batches": total_batches,
            "batch_size": len(batch_indices),
            "loss": results["loss"],
            "accuracy": results["accuracy"],
            "samples_processed": results["samples_processed"],
            "training_time": results["training_time"],
            "model_state": model_state_serializable,
            "status": "completed"
        }
        
        logger.info(f"Batch {batch_id} completado - Loss: {results['loss']:.4f}, Acc: {results['accuracy']:.4f}")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error en train_batch: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_endpoint():
    ensure_initialized()
    """Evalúa el modelo en el conjunto de test"""
    try:
        data = request.get_json() or {}
        model_state = data.get('model_state')
        
        # Cargar estado del modelo si se proporciona
        if model_state:
            # Convertir de lista a tensor
            state_dict = {}
            for key, tensor_list in model_state.items():
                state_dict[key] = torch.tensor(tensor_list)
            model.load_state_dict(state_dict)
        
        test_loader = batch_manager.get_test_loader()
        test_loss, test_acc = evaluate_batch(model, test_loader, torch.device("cpu"))
        
        return jsonify({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "samples_evaluated": len(batch_manager.test_ds)
        }), 200
        
    except Exception as e:
        logger.error(f"Error en evaluate: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ====== Inicialización y Servidor ======
if __name__ == '__main__':
    # Inicializar componentes
    initialize_components()
    
    # Obtener puerto de las variables de entorno (Render usa PORT)
    port = int(os.environ.get('PORT', 3001))
    
    # Ejecutar servidor
    app.run(host='0.0.0.0', port=port, debug=False)