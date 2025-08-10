
# Entrenamiento local (validación)

## 1) Crear entorno
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## 2) Entrenar rápido
```bash
python train_local.py --epochs 2 --batch-size 128
```
Flags útiles: `--cpu` para forzar CPU, `--workers 4` para acelerar *dataloading*, `--data ~/.torch-datasets` para cache local.

## 3) Resultados
- Checkpoint mejor modelo en `./checkpoints/best.pt`.
- Métricas por época en consola.

## 4) Paridad con el sistema distribuido
- Arquitectura de red **SmallCIFAR** coincide con la del MVP en Render.
- Hiperparámetros por defecto compatibles (`lr=0.1, momentum=0.9, wd=5e-4, bs=128`).

## 5) Próximo paso
Una vez valides que loss baja y accuracy sube (val_acc ~0.55–0.65 en pocas épocas con esta CNN sencilla), pasamos a desplegar el **parámetro-servidor** en Render y a conectar *workers*.
