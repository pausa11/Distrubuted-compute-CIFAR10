import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import matplotlib.pyplot as plt

modelo = np.load('modelo_entrenado.npz')
W1, b1, W2, b2 = modelo['W1'], modelo['b1'], modelo['W2'], modelo['b2']

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def predecir_digito(img):
    img = img.convert("L")
    img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    # Redimensionar a 20x20 manteniendo proporción
    max_dim = max(img.size)
    scale = 20.0 / max_dim
    new_size = [int(x * scale) for x in img.size]
    img = img.resize(new_size, Image.LANCZOS)

    # Crear lienzo 28x28 y centrar
    background = Image.new("L", (28, 28), color=0)
    offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    background.paste(img, offset)

    # Aplanar y normalizar
    arr = np.asarray(background).astype(np.float32) / 255.0
    arr = arr.reshape(1, 784)

    # Forward pass
    z1 = arr.dot(W1.T) + b1
    a1 = relu(z1)
    z2 = a1.dot(W2.T) + b2
    a2 = softmax(z2)

    return np.argmax(a2), a2

def visualizar_preprocesamiento(img):
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle("Visualización del preprocesamiento", fontsize=14)

    img1 = img.convert("L")
    axes[0].imshow(img1, cmap='gray')
    axes[0].set_title("Original (gris)")
    axes[0].axis("off")

    img2 = ImageOps.invert(img1)
    axes[1].imshow(img2, cmap='gray')
    axes[1].set_title("Invertida")
    axes[1].axis("off")

    bbox = img2.getbbox()
    img3 = img2.crop(bbox) if bbox else img2
    axes[2].imshow(img3, cmap='gray')
    axes[2].set_title("Recortada")
    axes[2].axis("off")

    max_dim = max(img3.size)
    scale = 20.0 / max_dim
    new_size = [int(x * scale) for x in img3.size]
    img4 = img3.resize(new_size, Image.LANCZOS)

    final = Image.new("L", (28, 28), color=0)
    offset = ((28 - new_size[0]) // 2, (28 - new_size[1]) // 2)
    final.paste(img4, offset)

    axes[3].imshow(final, cmap='gray')
    axes[3].set_title("Final (28x28)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()

class DigitoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de Dígitos 🧐")
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.btn_predecir = tk.Button(self.button_frame, text="Predecir", command=self.predecir)
        self.btn_predecir.pack(side=tk.LEFT, padx=5)

        self.btn_borrar = tk.Button(self.button_frame, text="Borrar", command=self.limpiar)
        self.btn_borrar.pack(side=tk.LEFT, padx=5)

        self.label_resultado = tk.Label(root, text="Dibuja un número y presiona 'Predecir'")
        self.label_resultado.pack(pady=5)

        self.imagen = Image.new("RGB", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.imagen)

        self.canvas.bind("<B1-Motion>", self.dibujar)

    def dibujar(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='black', outline='black')
        self.draw.ellipse((x - r, y - r, x + r, y + r), fill='black')

    def predecir(self):
        visualizar_preprocesamiento(self.imagen)
        numero, probabilidades = predecir_digito(self.imagen)
        self.label_resultado.config(text=f"Predicción: {numero} | Probabilidades: {np.round(probabilidades, 2)}")

    def limpiar(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")
        self.label_resultado.config(text="Dibuja un número y presiona 'Predecir'")

if __name__ == '__main__':
    root = tk.Tk()
    app = DigitoApp(root)
    root.mainloop()
