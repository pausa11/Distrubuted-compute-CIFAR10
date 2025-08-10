import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

TRAIN_CSV = Path('./train.csv')
TEST_CSV  = Path('./test.csv')

def categoryVector(y):
    num_classes = 10
    out = np.zeros((y.size, num_classes), dtype=np.float32)
    out[np.arange(y.size), y] = 1.0
    return out

def relu(z):
    return np.maximum(0, z)

def drelu(z):
    return (z > 0).astype(z.dtype)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def log_loss(y_true, y_pred):
    eps = 1e-15
    p = np.clip(y_pred, eps, 1-eps)
    return -np.mean(np.sum(y_true * np.log(p), axis=1))

def accuracy(pred_probs, y_true_vec):
    return np.mean(pred_probs.argmax(1) == y_true_vec.argmax(1))


# Carga y preprocesamiento
df = pd.read_csv(TRAIN_CSV)
X = df.drop('label', axis=1).values.astype(np.float32) / 255.0
y_num = df['label'].values
Y = categoryVector(y_num)

# Arquitectura y parámetros
neuronIn      = 784
neuronsHidden = 20
neuronOut     = 10
learningRate  = 0.1
iterations    = 100
rng           = np.random.default_rng(33)

# Inicialización de pesos
W1 = rng.uniform(-0.5, 0.5, size=(neuronsHidden, neuronIn)).astype(np.float32)
b1 = np.zeros((1, neuronsHidden), dtype=np.float32)
W2 = rng.uniform(-0.5, 0.5, size=(neuronOut, neuronsHidden)).astype(np.float32)
b2 = np.zeros((1, neuronOut), dtype=np.float32)

loss_history = []

# Entrenamiento sin batches
for it in range(1, iterations + 1):
    # Forward pass (todo el conjunto)
    Z1 = X.dot(W1.T) + b1          # (N, neuronsHidden)
    A1 = relu(Z1)                  # (N, neuronsHidden)
    Z2 = A1.dot(W2.T) + b2         # (N, neuronOut)
    A2 = softmax(Z2)               # (N, neuronOut)

    # Backpropagation (gradientes de full-batch)
    N = X.shape[0]
    dZ2 = (A2 - Y) / N             # (N, neuronOut)
    dW2 = dZ2.T.dot(A1)            # (neuronOut, neuronsHidden)
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2.dot(W2)              # (N, neuronsHidden)
    dZ1 = dA1 * drelu(Z1)          # (N, neuronsHidden)
    dW1 = dZ1.T.dot(X)             # (neuronsHidden, neuronIn)
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Actualización de parámetros
    W2 -= learningRate * dW2
    b2 -= learningRate * db2
    W1 -= learningRate * dW1
    b1 -= learningRate * db1

    # Registrar pérdida
    loss = log_loss(Y, A2)
    loss_history.append(loss)

# Gráfica de la pérdida
plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.title("Pérdida del entrenamiento (full-batch)")
plt.xlabel("Iteración")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Predicción sobre test.csv
testData = pd.read_csv(TEST_CSV).values.astype(np.float32) / 255.0
Z1_t = testData.dot(W1.T) + b1
A1_t = relu(Z1_t)
pred_test = softmax(A1_t.dot(W2.T) + b2).argmax(axis=1)

# Guardar submission
sub = pd.DataFrame({
    'ImageId': np.arange(1, len(pred_test) + 1),
    'Label': pred_test
})
sub.to_csv('submission.csv', index=False)

# Guardar modelo
np.savez('modelo_entrenado_fullbatch.npz', W1=W1, b1=b1, W2=W2, b2=b2)

# Exactitud final en entrenamiento
Z1_f = X.dot(W1.T) + b1
A1_f = relu(Z1_f)
A2_f = softmax(A1_f.dot(W2.T) + b2)
print("Exactitud final sobre entrenamiento:", accuracy(A2_f, Y))
