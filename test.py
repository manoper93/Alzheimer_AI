from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import pickle
import os
import torch
import tkinter as tk
from tkinter import filedialog

# 🔹 Verificar GPUs disponíveis
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ Teste usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ Nenhuma GPU disponível, usando CPU.")

# 🔹 Verificar se os arquivos necessários existem
if not os.path.exists("class_names.pkl"):
    raise FileNotFoundError("❌ ERRO: 'class_names.pkl' não foi encontrado. Execute 'train.py' primeiro!")

if not os.path.exists("modelo_alzheimer.keras"):
    raise FileNotFoundError("❌ ERRO: 'modelo_alzheimer.keras' não foi encontrado. Execute 'train.py' primeiro!")

# 🔹 Carregar class_names salvo no treino
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

print("✅ Lista de classes carregada com sucesso!")

# 🔹 Carregar o modelo treinado no formato Keras
model = load_model("modelo_alzheimer.keras")

# 🔹 Abrir o explorador de arquivos para selecionar a imagem
root = tk.Tk()
root.withdraw()  # Esconde a janela principal
IMG_PATH = filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

if not IMG_PATH:
    raise FileNotFoundError("❌ ERRO: Nenhuma imagem selecionada!")

# 🔹 Pré-processar a imagem
img = image.load_img(IMG_PATH, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalizar

# 🔹 Fazer a previsão
predictions = model.predict(img_array)
predicted_class = class_names[np.argmax(predictions)]  # 🔹 Correção!

print("🔍 Classe prevista:", predicted_class)
