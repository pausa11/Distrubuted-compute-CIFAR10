
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
import numpy as np

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
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)
normalize = T.Normalize(CIFAR10_MEAN, CIFAR10_STD)
to_tensor = T.ToTensor()

def preprocess_pil(pil_img):
    pil_img = pil_img.convert("RGB")
    pil_img_resized = pil_img.resize((32, 32), Image.BICUBIC)
    x = to_tensor(pil_img_resized)
    x = normalize(x)
    return x, pil_img_resized

CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def topk_from_logits(logits, k=5):
    probs = torch.softmax(logits, dim=-1).squeeze(0)
    vals, idxs = torch.topk(probs, k)
    return [(CLASSES[i], float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]

class CIFARApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("CIFAR-10 Inference (Tk)")
        self.geometry("880x600")
        self.minsize(820, 560)

        self.model = None
        self.device = torch.device("cpu")
        self.current_pil = None
        self.current_tensor = None

        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(top, text="Checkpoint (.pt):").pack(side=tk.LEFT)
        self.ckpt_var = tk.StringVar(value="checkpoints/best.pt")
        self.ckpt_entry = ttk.Entry(top, textvariable=self.ckpt_var, width=60)
        self.ckpt_entry.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Buscar...", command=self.browse_ckpt).pack(side=tk.LEFT, padx=4)
        ttk.Button(top, text="Cargar modelo", command=self.load_model).pack(side=tk.LEFT, padx=4)

        # badge de estado del modelo
        self.model_badge = ttk.Label(top, text="Sin modelo", foreground="#cc4444")
        self.model_badge.pack(side=tk.LEFT, padx=10)

        mid = ttk.Frame(self, padding=(10, 0, 10, 10))
        mid.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(mid, text="Abrir imagen...", command=self.open_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(mid, text="Usar random de CIFAR-10", command=self.load_random_cifar).pack(side=tk.LEFT, padx=4)
        ttk.Button(mid, text="Predecir", command=self.predict).pack(side=tk.LEFT, padx=4)

        main = ttk.Frame(self, padding=10)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        img_frame = ttk.LabelFrame(main, text="Imagen")
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,10))
        self.canvas = tk.Canvas(img_frame, bg="#222222", width=360, height=360, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        res_frame = ttk.LabelFrame(main, text="Resultados (Top-5)")
        res_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.results = tk.StringVar(value="Carga un modelo y una imagen para predecir.")
        self.results_label = ttk.Label(res_frame, textvariable=self.results, justify=tk.LEFT, anchor="nw")
        self.results_label.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.status = tk.StringVar(value="Listo.")
        status_bar = ttk.Label(self, textvariable=self.status, relief=tk.SUNKEN, anchor="w")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        try:
            self.style = ttk.Style(self)
            if "vista" in self.style.theme_names():
                self.style.theme_use("vista")
        except Exception:
            pass

    def browse_ckpt(self):
        path = filedialog.askopenfilename(
            title="Seleccionar checkpoint .pt",
            filetypes=[("PyTorch checkpoint", "*.pt *.pth"), ("Todos", "*.*")]
        )
        if path:
            self.ckpt_var.set(path)
            self.load_model()

    def _try_load_torchscript(self, path):
        try:
            model = torch.jit.load(path, map_location="cpu")
            model.eval()
            return model
        except Exception:
            return None

    def _set_model_loaded_ui(self, ok: bool, path: str = ""):
        if ok:
            self.model_badge.configure(text="Modelo cargado", foreground="#3aa657")
            self.status.set(f"Modelo cargado: {os.path.basename(path)}")
            self.title(f"CIFAR-10 Inference (Tk) — {os.path.basename(path)}")
        else:
            self.model_badge.configure(text="Sin modelo", foreground="#cc4444")
            self.status.set("No hay modelo cargado.")

    def load_model(self):
        path = self.ckpt_var.get().strip()
        if not os.path.exists(path):
            messagebox.showerror("Error", f"No existe el archivo: {path}")
            self._set_model_loaded_ui(False)
            return

        model = self._try_load_torchscript(path)
        if model is not None:
            self.model = model.to(self.device)
            self._set_model_loaded_ui(True, path)
            return

        try:
            obj = torch.load(path, map_location="cpu")
            if isinstance(obj, dict):
                candidates = ["model_state", "state_dict", "model", "net"]
                state = None
                for k in candidates:
                    if k in obj and isinstance(obj[k], dict):
                        state = obj[k]
                        break
                if state is None:
                    looks_like_state = all(isinstance(v, torch.Tensor) for v in obj.values())
                    if looks_like_state:
                        state = obj
                if state is None:
                    raise RuntimeError("No encontré un state_dict válido ('model_state', 'state_dict', 'model' o 'net').")
                model = SmallCIFAR()
                model.load_state_dict(state, strict=False)
                model.eval()
                self.model = model.to(self.device)
                self._set_model_loaded_ui(True, path)
            else:
                raise RuntimeError("Formato desconocido del checkpoint.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el modelo:\n{e}")
            self.model = None
            self._set_model_loaded_ui(False)

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("Todos", "*.*")]
        )
        if not path:
            return
        try:
            from PIL import Image
            pil = Image.open(path).convert("RGB")
            tensor, pil_resized = preprocess_pil(pil)
            self.set_current_image(pil_resized, tensor)
            self.status.set(f"Imagen cargada: {os.path.basename(path)} (32x32)")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo abrir la imagen:\n{e}")

    def load_random_cifar(self):
        try:
            from PIL import Image
            data_root = os.path.expanduser("~/.torch-datasets")
            ds = CIFAR10(root=data_root, train=False, download=True)
            idx = np.random.randint(0, len(ds))
            pil, label = ds.data[idx], ds.targets[idx]
            pil = Image.fromarray(pil, mode="RGB")
            tensor, pil_resized = preprocess_pil(pil)
            self.set_current_image(pil_resized, tensor)
            self.status.set(f"Imagen aleatoria de CIFAR-10 cargada (label real: {CLASSES[label]}).")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar CIFAR-10:\n{e}")

    def set_current_image(self, pil_resized, tensor):
        self.current_pil = pil_resized
        self.current_tensor = tensor.unsqueeze(0)
        disp = pil_resized.resize((360, 360), Image.NEAREST)
        self.tk_img = ImageTk.PhotoImage(disp)
        self.canvas.delete("all")
        self.canvas.create_image(10, 10, image=self.tk_img, anchor="nw")

    def predict(self):
        # Intento auto-cargar si hay ruta y aún no hay modelo
        if self.model is None:
            path = self.ckpt_var.get().strip()
            if path:
                self.load_model()
        if self.model is None:
            messagebox.showwarning("Atención", "Primero carga un modelo (.pt).")
            return
        if self.current_tensor is None:
            messagebox.showwarning("Atención", "Primero carga una imagen o usa una aleatoria de CIFAR-10.")
            return
        try:
            with torch.no_grad():
                logits = self.model(self.current_tensor.to(self.device))
                top5 = topk_from_logits(logits, k=5)
                lines = [f"{i+1}. {cls}: {prob*100:.2f}%" for i,(cls,prob) in enumerate(top5)]
                self.results.set("Top-5 predicciones:\n" + "\n".join(lines))
                self.status.set("Predicción realizada.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo predecir:\n{e}")

def main():
    app = CIFARApp()
    app.mainloop()

if __name__ == "__main__":
    main()
