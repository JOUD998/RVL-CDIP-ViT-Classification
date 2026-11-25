
import json
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import vit_b_16, ViT_B_16_Weights

from scripts.data_pipeline.prepare_vit_dataset import train_loader, validation_loader
from scripts.lora.lora_attention import LoRAMultiheadAttention

# Hyperparameter für die LoRA-Anpassung und das Training
RANK = 8
ALPHA = 16
LEARNING_RATE = 0.0001
NUM_EPOCHS = 3

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device, flush=True)

# Vortrainiertes ViT-B/16-Modell laden
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

# Ursprüngliche Modellparameter einfrieren
for param in model.parameters():
    param.requires_grad = False

# LoRA in die Attention-Schichten aller 12 Encoder-Blöcke integrieren
for layer in model.encoder.layers:
    layer.self_attention = LoRAMultiheadAttention(
        layer.self_attention,
        rank=RANK,
        alpha=ALPHA
    )

# Trainierbaren Klassifikationskopf für 16 Dokumentklassen einsetzen
model.heads.head = nn.Linear(
    model.heads.head.in_features, 16
)

model = model.to(device)

# Evaluierungsmodus beibehalten; die Gradientenberechnung für LoRA
# und den Klassifikationskopf bleibt trotzdem aktiviert
model.eval()

# Ausschließlich trainierbare Parameter an den Optimierer übergeben
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad),
    lr=LEARNING_RATE
)

criterion = nn.CrossEntropyLoss()

checkpoint_dir = (
    Path(__file__).resolve().parents[2]
    / "checkpoints"
    / f"vit_lora_r{RANK}"
)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

results_dir = (
    Path(__file__).resolve().parents[2]
    / "results"
    / f"vit_lora_r{RANK}"
)
results_dir.mkdir(parents=True, exist_ok=True)

history = []

best_val_accuracy = 0.0

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("Trainable parameters:", trainable_params, flush=True)

for epoch in range(NUM_EPOCHS):

    running_loss = 0.0

    # Trainingsdurchlauf mit Gradientenberechnung und Parameteraktualisierung
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Trainingsfortschritt alle 200 Batches ausgeben
        running_loss += loss.item() * images.size(0)

        if (batch_idx + 1) % 200 == 0:
            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                f"Loss: {loss.item():.4f}",
                flush=True
            )

    train_loss = running_loss / len(train_loader.dataset)

    val_loss = 0.0
    correct = 0
    total = 0

    # Validierung ohne Gradientenberechnung nach jeder Epoche
    with torch.no_grad():
        for images, labels in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_accuracy = 100 * correct / total

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS}\n"
        f"Train Loss: {train_loss:.4f}\n"
        f"Validation Loss: {val_loss:.4f}\n"
        f"Validation Accuracy: {val_accuracy:.2f}%\n",
        flush=True
    )

    # Aktuellen Trainingsstand einschließlich Optimiererzustand speichern
    torch.save(
        {
            "epoch": epoch + 1,
            "rank": RANK,
            "alpha": ALPHA,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": val_accuracy
        },
        checkpoint_dir / "latest.pth"
    )

    # Bestes Modell anhand der bisherigen Validierungsgenauigkeit auswählen
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            checkpoint_dir / "best.pth"
        )

        print("New best LoRA model saved!", flush=True)

    # Trainings- und Validierungsmetriken für jede Epoche protokollieren
    history.append({
        "epoch": epoch + 1,
        "train_loss": round(train_loss, 4),
        "validation_loss": round(val_loss, 4),
        "validation_accuracy_percent": round(val_accuracy, 2)
    })

    with open(results_dir / "training_history.json", "w") as f:
        json.dump({
            "model": "ViT-B/16 LoRA",
            "rank": RANK,
            "alpha": ALPHA,
            "history": history
        }, f, indent=4)

print("Training finished!", flush=True)
