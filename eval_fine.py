from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Diretório com dados de teste
test_dir = "fine_tuning_split/test"
img_size = (128, 128)
batch_size = 32

# Carregar dataset de teste (sem normalizar ainda)
test_dataset_raw = image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    shuffle=False
)

# Guardar classes antes do .map
class_names = test_dataset_raw.class_names

# Normalizar imagens
test_dataset = test_dataset_raw.map(lambda x, y: (x / 255.0, y))

# Carregar modelo ajustado
model = load_model("modelo_alzheimer_finetuned.keras")

# Previsões
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.concatenate([y for x, y in test_dataset], axis=0)

# Relatório de classificação
report = classification_report(y_true, y_pred, target_names=class_names, digits=2)
print("Classification Report:\n", report)

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
