from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image_dataset_from_directory

# 🔁 Diretórios
finetune_dir = "fine_tuning_split/finetune"  # Caminho relativo
img_size = (128, 128)
batch_size = 32

# 🔁 Carregar dados de fine-tuning
finetune_dataset = image_dataset_from_directory(
    finetune_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=True
)

# 🔁 Normalizar imagens
finetune_dataset = finetune_dataset.map(lambda x, y: (x / 255.0, y))

# 🔁 Carregar modelo existente
model = load_model("modelo_alzheimer.keras")

# 🔁 Tornar todas as camadas treináveis
for layer in model.layers:
    layer.trainable = True

# 🔁 Recompilar com learning rate mais baixo
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 🔁 Callback para evitar overfitting
early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)

# 🔁 Treinar
history = model.fit(
    finetune_dataset,
    epochs=10,
    callbacks=[early_stop]
)

# 🔁 Guardar novo modelo
model.save("modelo_alzheimer_finetuned.keras")
print("✅ Fine-tuning concluído com sucesso.")
