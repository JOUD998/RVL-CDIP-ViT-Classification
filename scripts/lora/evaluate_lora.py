
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import vit_b_16

from scripts.data_pipeline.prepare_vit_dataset import (
    validation_loader,
    test_loader
)

from scripts.lora.lora_attention import LoRAMultiheadAttention

RANK = 4
ALPHA = 8

# Datensatzteil für die Auswertung auswählen
EVALUATION_SPLIT = "test"  # "validation" oder "test"

if EVALUATION_SPLIT == "test":
    data_loader = test_loader
elif EVALUATION_SPLIT == "validation":
    data_loader = validation_loader
else:
    raise ValueError("Invalid evaluation split")

class_names = data_loader.dataset.classes

# Identische Klassenreihenfolge für die korrekte Zuordnung der Vorhersagen sicherstellen
assert class_names == validation_loader.dataset.classes, (
    "Class order mismatch between datasets!"
)

project_dir = Path(__file__).resolve().parents[2]

checkpoint_path = (
    project_dir / "checkpoints"
    / f"vit_lora_r{RANK}" / "best.pth"
)

results_dir = (
    project_dir
    / "results"
    / f"vit_lora_r{RANK}"
    / EVALUATION_SPLIT
)

results_dir.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device, flush=True)

# Modellarchitektur entsprechend der Trainingskonfiguration rekonstruieren
model = vit_b_16(weights=None)

for layer in model.encoder.layers:
    layer.self_attention = LoRAMultiheadAttention(
        layer.self_attention,
        rank=RANK,
        alpha=ALPHA
    )

model.heads.head = nn.Linear(
    model.heads.head.in_features,
    len(class_names)
)

# Gespeicherte Modellgewichte laden und Evaluierungsmodus aktivieren
state_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Konfusionsmatrix: Zeilen enthalten die tatsächlichen, Spalten die vorhergesagten Klassen
num_classes = len(class_names)
confusion_matrix = torch.zeros(
    num_classes, num_classes, dtype=torch.int64
)

correct = 0
total = 0
total_loss = 0.0

criterion = nn.CrossEntropyLoss()

# Auswertung ohne Gradientenberechnung durchführen
with torch.no_grad():
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        predictions = outputs.argmax(dim=1)

        # Verlust über alle Bilder gewichten und korrekte Vorhersagen zählen
        total_loss += loss.item() * images.size(0)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        for true_label, predicted_label in zip(
            labels.cpu().tolist(),
            predictions.cpu().tolist()
        ):
            confusion_matrix[true_label, predicted_label] += 1

accuracy = 100 * correct / total
evaluation_loss = total_loss / total

print(f"\n{EVALUATION_SPLIT.title()} Accuracy: {accuracy:.2f}%")
print(f"{EVALUATION_SPLIT.title()} Loss: {evaluation_loss:.4f}")

# Klassenspezifische Genauigkeit aus der Konfusionsmatrix berechnen
per_class = {}

for i, class_name in enumerate(class_names):
    class_total = confusion_matrix[i].sum().item()
    class_correct = confusion_matrix[i, i].item()

    class_accuracy = (
        100 * class_correct / class_total
        if class_total > 0 else 0.0
    )

    per_class[class_name] = {
        "total": class_total,
        "correct": class_correct,
        "accuracy_percent": round(class_accuracy, 2)
    }

    print(f"{class_name}: {class_accuracy:.2f}%")

# Gesamtmetriken und Ergebnisse pro Klasse zusammenfassen
metrics = {
    "model": "ViT-B/16 LoRA",
    "rank": RANK,
    "alpha": ALPHA,
    "evaluation_split": EVALUATION_SPLIT,
    "total_images": total,
    "correct_predictions": correct,
    "accuracy_percent": round(accuracy, 2),
    "evaluation_loss": round(evaluation_loss, 4),
    "per_class": per_class
}

# Ergebnisse als JSON und Konfusionsmatrix als CSV exportieren
with open(results_dir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open(
    results_dir / "confusion_matrix.csv",
    "w",
    newline=""
) as f:
    writer = csv.writer(f)
    writer.writerow(["true/predicted"] + class_names)

    for i, class_name in enumerate(class_names):
        writer.writerow(
            [class_name] + confusion_matrix[i].tolist()
        )

print("\nResults saved to:", results_dir)
