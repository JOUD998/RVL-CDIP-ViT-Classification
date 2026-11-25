
import torch
import torch.nn as nn
from pathlib import Path
from torchvision.models import vit_b_16, ViT_B_16_Weights
from scripts.data_pipeline.prepare_vit_dataset import train_loader, validation_loader

# Apple MPS verwenden, falls verfügbar, andernfalls CPU
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device:", device, flush=True)

# Vortrainiertes ViT-B/16-Modell laden
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

# Vortrainierte Parameter des Backbones einfrieren
for param in model.parameters():
    param.requires_grad = False

# Trainierbaren Klassifikationskopf für die 16 Dokumentklassen einsetzen
model.heads.head = nn.Linear(
    model.heads.head.in_features, 16
)

model = model.to(device)

# Linear Probing: Backbone im Evaluierungsmodus belassen
# Die Gradientenberechnung für den Klassifikationskopf bleibt aktiviert
model.eval()

# Nur die Parameter des Klassifikationskopfes optimieren
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.heads.head.parameters(),
    lr=0.001
)

num_epochs = 3
best_val_accuracy = 0.0

checkpoint_dir = (
    Path(__file__).resolve().parent.parent / "checkpoints"
)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(num_epochs):
    running_loss = 0.0

    # Trainingsdurchlauf mit Gradientenberechnung und Parameteraktualisierung
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Verlust nach Batchgröße gewichten und Trainingsfortschritt ausgeben
        running_loss += loss.item() * images.size(0)

        if (batch_idx + 1) % 200 == 0:
            progress = 100 * (batch_idx + 1) / len(train_loader)
            avg_loss = running_loss / (
                (batch_idx + 1) * train_loader.batch_size
            )
            print(
                f"Epoch {epoch + 1} | "
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"({progress:.0f}%) | "
                f"Loss: {avg_loss:.4f}",
                flush=True
            )

    train_loss = running_loss / len(train_loader.dataset)

    # Validierung nach jeder Epoche ohne Gradientenberechnung
    val_loss = 0.0
    correct = 0
    total = 0

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

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%\n")

    # Aktuellen Trainingsstand einschließlich Optimiererzustand speichern
    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": val_accuracy,
        },
        checkpoint_dir / "vit_latest.pth"
    )

    # Bestes Modell anhand der Validierungsgenauigkeit speichern
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            checkpoint_dir / "vit_best.pth"
        )

        print("New best ViT model saved!", flush=True)

print("Training finished!")
