import os
import random
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from flask import Flask, request, jsonify
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuración para optimizar memoria
torch.set_num_threads(1)  # Limitar hilos para reducir memoria
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TORCH_HOME'] = '/tmp/torch-cache'

app = Flask(__name__)

# Variables globales que se inicializan bajo demanda
_model = None
_batch_manager = None

def cleanup_memory():
    """Limpia memoria y cache de PyTorch"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def get_model():
    """Lazy loading del modelo"""
    global _model
    if _model is None:
        _model = SmallCIFAR().to(torch.device("cpu"))
        # Usar half precision para reducir memoria (si es compatible)
        if hasattr(torch, 'float16'):
            _model = _model.half()
        logger.info("Modelo cargado")
    return _model

def get_batch_manager():
    """Lazy loading del batch manager"""
    global _batch_manager
    if _batch_manager is None:
        data_root = os.environ.get('DATA_ROOT', '/tmp/torch-datasets')
        _batch_manager = BatchManager(data_root)
        logger.info("BatchManager cargado")
    return _batch_manager

# ====== Modelo optimizado ======
class SmallCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Reducir canales para menos memoria
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)  # 32 -> 16
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)  # 64 -> 32
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)  # 128 -> 64
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 256 -> 128
        self.fc2 = nn.Linear(128, num_classes)
        
        # Inicialización eficiente
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 16x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 32x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 64x4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ====== Dataset y Batch Management optimizado ======
class BatchManager:
    def __init__(self, data_root: str, batch_size: int = 64):  # Reducir batch size
        self.data_root = data_root
        self.batch_size = batch_size
        self.device = torch.device("cpu")
        
        # NO cargar datasets automáticamente
        self._train_ds = None
        self._test_ds = None
        
        # Crear directorio si no existe
        Path(data_root).mkdir(parents=True, exist_ok=True)
        
    def _get_transforms(self):
        """Transformaciones optimizadas"""
        # Usar mean/std aproximados para reducir cálculos
        normalize = torch.nn.functional.normalize
        
        train_tf = torch.nn.Sequential(
            # Transformaciones mínimas para reducir memoria
        )
        
        return train_tf, train_tf  # Usar las mismas para train y test
        
    def _load_datasets(self):
        """Carga los datasets solo cuando se necesiten"""
        if self._train_ds is None:
            # Importar solo cuando se necesite
            import torchvision
            import torchvision.transforms as T
            
            # Transformaciones mínimas
            simple_transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalización simple
            ])
            
            try:
                self._train_ds = torchvision.datasets.CIFAR10(
                    root=self.data_root, 
                    train=True, 
                    download=True, 
                    transform=simple_transform
                )
                self._test_ds = torchvision.datasets.CIFAR10(
                    root=self.data_root, 
                    train=False, 
                    download=True, 
                    transform=simple_transform
                )
                logger.info(f"Datasets cargados: {len(self._train_ds)} train, {len(self._test_ds)} test")
            except Exception as e:
                logger.error(f"Error cargando datasets: {e}")
                raise
            
            # Limpar memoria después de cargar
            cleanup_memory()
    
    @property
    def train_ds(self):
        if self._train_ds is None:
            self._load_datasets()
        return self._train_ds
    
    @property 
    def test_ds(self):
        if self._test_ds is None:
            self._load_datasets()
        return self._test_ds
    
    def get_batch_loader(self, batch_indices: list) -> DataLoader:
        """Crea un DataLoader optimizado"""
        subset = Subset(self.train_ds, batch_indices)
        return DataLoader(
            subset, 
            batch_size=min(self.batch_size, len(batch_indices)),
            shuffle=False, 
            num_workers=0,  # Sin workers paralelos para reducir memoria
            pin_memory=False,  # Desactivar pin_memory
            persistent_workers=False
        )
    
    def get_test_loader(self, max_samples: int = 1000) -> DataLoader:
        """Retorna un DataLoader de test limitado"""
        # Limitar muestras de test para reducir memoria
        indices = list(range(min(len(self.test_ds), max_samples)))
        subset = Subset(self.test_ds, indices)
        return DataLoader(
            subset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=False
        )
    
    def get_batch_indices(self, batch_id: int, total_batches: int) -> list:
        """Calcula los índices para un batch específico"""
        total_samples = len(self.train_ds)
        samples_per_batch = total_samples // total_batches
        
        start_idx = batch_id * samples_per_batch
        end_idx = (batch_id + 1) * samples_per_batch if batch_id < total_batches - 1 else total_samples
        
        return list(range(start_idx, end_idx))

# ====== Training Utils optimizado ======
def set_seed(seed: int = 1337):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@torch.no_grad()
def evaluate_batch(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """Evalúa el modelo en un batch específico - optimizado"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    
    try:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            # Usar autocast si está disponible para reducir memoria
            if hasattr(torch, 'autocast'):
                with torch.autocast(device_type='cpu', dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
                
            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Limpiar memoria después de cada batch
            del images, labels, logits
            cleanup_memory()
    
    except Exception as e:
        logger.error(f"Error en evaluación: {e}")
        raise
    
    return total_loss / total if total > 0 else 0.0, correct / total if total > 0 else 0.0

def train_batch(
    model: nn.Module, 
    batch_loader: DataLoader, 
    optimizer: optim.Optimizer, 
    device: torch.device,
    model_state: Optional[Dict] = None
) -> Dict[str, Any]:
    """Entrena el modelo en un batch específico - optimizado"""
    
    # Cargar estado del modelo si se proporciona
    if model_state:
        try:
            state_dict = {}
            for key, tensor_data in model_state.items():
                if isinstance(tensor_data, list):
                    state_dict[key] = torch.tensor(tensor_data)
                else:
                    state_dict[key] = tensor_data
            model.load_state_dict(state_dict)
        except Exception as e:
            logger.warning(f"Error cargando estado del modelo: {e}")
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    try:
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
            
            # Limpiar memoria después de cada batch
            del images, labels, logits, loss
            cleanup_memory()
    
    except Exception as e:
        logger.error(f"Error en entrenamiento: {e}")
        raise
    
    training_time = time.time() - start_time
    
    return {
        "loss": total_loss / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
        "samples_processed": total,
        "training_time": training_time,
        "model_state": model.state_dict()
    }

# ====== API Endpoints ======
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - sin inicialización"""
    return jsonify({
        "status": "healthy", 
        "message": "Service is running",
        "initialized": _model is not None and _batch_manager is not None
    }), 200

@app.route('/info', methods=['GET'])
def get_info():
    """Información del nodo - inicialización bajo demanda"""
    try:
        model = get_model()
        batch_manager = get_batch_manager()
        
        return jsonify({
            "device": "cpu",
            "model_params": sum(p.numel() for p in model.parameters()),
            "training_samples": len(batch_manager.train_ds),
            "test_samples": len(batch_manager.test_ds),
            "memory_info": {
                "allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "cached": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0
            }
        }), 200
    except Exception as e:
        logger.error(f"Error en info: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/train_batch', methods=['POST'])
def train_batch_endpoint():
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
        lr = data.get('lr', 0.01)  # LR más bajo para estabilidad
        momentum = data.get('momentum', 0.9)
        weight_decay = data.get('weight_decay', 1e-4)
        seed = data.get('seed', 1337)
        model_state = data.get('model_state')
        
        # Validar rangos
        if batch_id < 0 or batch_id >= total_batches:
            return jsonify({"error": "batch_id out of range"}), 400
        
        # Configurar seed
        set_seed(seed)
        
        # Inicializar componentes bajo demanda
        model = get_model()
        batch_manager = get_batch_manager()
        
        # Obtener índices del batch
        batch_indices = batch_manager.get_batch_indices(batch_id, total_batches)
        batch_loader = batch_manager.get_batch_loader(batch_indices)
        
        # Configurar optimizer con parámetros optimizados
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
        
        # Serializar estado del modelo de forma eficiente
        model_state_serializable = {}
        for key, tensor in results["model_state"].items():
            # Convertir a float16 para reducir tamaño
            if tensor.dtype == torch.float32:
                tensor = tensor.half()
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
        
        # Limpiar memoria al final
        cleanup_memory()
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error en train_batch: {str(e)}")
        cleanup_memory()
        return jsonify({"error": str(e)}), 500

@app.route('/evaluate', methods=['POST'])
def evaluate_endpoint():
    """Evalúa el modelo en el conjunto de test"""
    try:
        data = request.get_json() or {}
        model_state = data.get('model_state')
        max_samples = data.get('max_samples', 1000)  # Limitar muestras
        
        # Inicializar componentes bajo demanda
        model = get_model()
        batch_manager = get_batch_manager()
        
        # Cargar estado del modelo si se proporciona
        if model_state:
            try:
                state_dict = {}
                for key, tensor_list in model_state.items():
                    state_dict[key] = torch.tensor(tensor_list)
                model.load_state_dict(state_dict)
            except Exception as e:
                logger.warning(f"Error cargando estado: {e}")
        
        test_loader = batch_manager.get_test_loader(max_samples)
        test_loss, test_acc = evaluate_batch(model, test_loader, torch.device("cpu"))
        
        cleanup_memory()
        
        return jsonify({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "samples_evaluated": min(len(batch_manager.test_ds), max_samples)
        }), 200
        
    except Exception as e:
        logger.error(f"Error en evaluate: {e}")
        cleanup_memory()
        return jsonify({"error": str(e)}), 500

@app.route('/cleanup', methods=['POST'])
def cleanup_endpoint():
    """Endpoint para limpiar memoria manualmente"""
    global _model, _batch_manager
    
    # Limpiar variables globales
    _model = None
    _batch_manager = None
    
    cleanup_memory()
    
    return jsonify({"status": "cleaned", "message": "Memory cleaned successfully"}), 200

# ====== Configuración del servidor ======
if __name__ == '__main__':
    # NO inicializar componentes automáticamente
    logger.info("Servidor iniciado - componentes se cargarán bajo demanda")
    
    # Obtener puerto de las variables de entorno
    port = int(os.environ.get('PORT', 3001))
    
    # Ejecutar servidor
    app.run(host='0.0.0.0', port=port, debug=False)