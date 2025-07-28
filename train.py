import os
import tensorflow as tf
import torch
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
import pickle

# 🔹 Verificar se há GPU disponível
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"✅ A usar GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("⚠️ A usar CPU.")

# 🔹 Parâmetros
DATASET_PATH = "Teste"
IMG_SIZE = (128, 128)
BATCH_SIZE = 40
EPOCHS = 20  # Apenas as novas épocas desejadas
MODEL_PATH = "modelo_alzheimer.keras"

# 🔹 Carregar dados
raw_train_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

raw_val_dataset = image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 🔹 Guardar nomes das classes
class_names = raw_train_dataset.class_names
with open("class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

# 🔹 Normalizar
normalization_layer = layers.Rescaling(1./255)
train_dataset = raw_train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = raw_val_dataset.map(lambda x, y: (normalization_layer(x), y))

# 🔹 Criar ou carregar modelo
if os.path.exists(MODEL_PATH):
    print("🔄 A carregar modelo existente...")
    model = load_model(MODEL_PATH)
else:
    print("🆕 A criar novo modelo...")
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

# 🔹 Continuar o treino
with tf.device("/GPU:0" if torch.cuda.is_available() else "/CPU:0"):
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS)

# 🔹 Guardar modelo atualizado
model.save(MODEL_PATH)
print(f"✅ Modelo guardado como '{MODEL_PATH}'")

# 🔹 Gráfico de treino
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (Continued)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig("grafico_accuracy.png")
plt.show()
