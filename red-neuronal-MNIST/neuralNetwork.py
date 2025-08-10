import numpy as np, pandas as pd, matplotlib.pyplot as plt
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

def accuracy(pred_probs, yCategoryVector):
    return np.mean(pred_probs.argmax(1) == yCategoryVector.argmax(1))

trainingData = pd.read_csv(TRAIN_CSV)
imageX = trainingData.drop('label', axis=1).values.astype(np.float32)       # pixeles
numberY = trainingData['label'].values                                      # etiquetas

imageX /= 255.0
yCategoryVector = categoryVector(numberY)  

neuronIn        = 784
neuronsHidden   = 20
neuronOut       = 10
learningRate    = 0.1
iterations      = 100
batchSize       = 128
rng             = np.random.default_rng(33)

W1 = rng.uniform(-0.5, 0.5, size=(neuronsHidden, neuronIn)).astype(np.float32)
b1 = np.zeros((1, neuronsHidden), dtype=np.float32)
W2 = rng.uniform(-0.5, 0.5, size=(neuronOut, neuronsHidden)).astype(np.float32)
b2 = np.zeros((1, neuronOut), dtype=np.float32)

loss_history = []

for it in range(1, iterations + 1):
    idx = rng.permutation(imageX.shape[0])
    imageX, yCategoryVector = imageX[idx], yCategoryVector[idx]
    
    for i in range(0, imageX.shape[0], batchSize):
        Xb = imageX[i:i+batchSize]
        yb = yCategoryVector[i:i+batchSize]
        B  = Xb.shape[0]

        Z1 = Xb.dot(W1.T) + b1
        A1 = relu(Z1)
        Z2 = A1.dot(W2.T) + b2
        A2 = softmax(Z2)

        dZ2 = (A2 - yb) / B
        dW2 = dZ2.T.dot(A1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dA1 = dZ2.dot(W2)
        dZ1 = dA1 * drelu(Z1)
        dW1 = dZ1.T.dot(Xb)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        W2 -= learningRate * dW2
        b2 -= learningRate * db2
        W1 -= learningRate * dW1
        b1 -= learningRate * db1

    #trackeamos pérdida del entrenamiento
    Z1 = imageX.dot(W1.T) + b1
    A1 = relu(Z1)
    A2 = softmax(A1.dot(W2.T) + b2)
    loss = log_loss(yCategoryVector, A2)
    loss_history.append(loss)

plt.figure(figsize=(6,4))
plt.plot(loss_history)
plt.title("Pérdida del entrenamiento")
plt.xlabel("Iteración"); plt.ylabel("Loss"); plt.grid(); plt.show()

# Predicción en test.csv
testData = pd.read_csv(TEST_CSV).astype(np.float32) / 255.0
Z1_t = testData.values.dot(W1.T) + b1
A1_t = relu(Z1_t)
pred_test = softmax(A1_t.dot(W2.T) + b2).argmax(1)

sub = pd.DataFrame({'ImageId': np.arange(1, len(pred_test)+1), 'Label': pred_test})
sub.to_csv('submission.csv', index=False)

np.savez('modelo_entrenado.npz', W1=W1, b1=b1, W2=W2, b2=b2)
print("Exactitud final sobre entrenamiento:", accuracy(A2, yCategoryVector))
