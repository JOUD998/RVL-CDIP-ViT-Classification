
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from scripts.data_pipeline.prepare_cnn_dataset import train_loader, validation_loader
from pathlib import Path

# Vortrainiertes ResNet18-Modell laden
model = resnet18(weights=ResNet18_Weights.DEFAULT)

# Vortrainierte Parameter einfrieren
for param in model.parameters():
    param.requires_grad = False

# Klassifikationsschicht für die 16 Dokumentklassen ersetzen
model.fc = nn.Linear(model.fc.in_features, 16)

# Verlustfunktion und Optimierer für die trainierbare Klassifikationsschicht
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)



# Evaluierungsmodus beibehalten, um die Batch-Normalization-Statistiken
# des eingefrorenen Backbones während des Trainings nicht zu verändern
model.eval()

num_epochs = 3

best_val_accuracy = 0.0

for epoch in range(num_epochs):

    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        # Vorwärtsdurchlauf und Verlustberechnung
        outputs = model(images)

        # Kreuzentropieverlust berechnen
        loss = criterion(outputs, labels)

        # Gradienten für die trainierbare Klassifikationsschicht berechnen
        loss.backward()

        # Modellparameter aktualisieren
        optimizer.step()

        # Verlust nach Batchgröße gewichten und Trainingsfortschritt ausgeben
        running_loss += loss.item() * images.size(0)
        if (batch_idx + 1) % 100 == 0:
            progress = (batch_idx + 1) / len(train_loader) * 100
            current_loss = running_loss / ((batch_idx + 1) * train_loader.batch_size)

            print(
                f"Batch {batch_idx + 1}/{len(train_loader)} "
                f"({progress:.0f}%) - Avg Loss: {current_loss:.4f}",
                flush=True
            )

    epoch_loss = running_loss / len(train_loader.dataset)

    print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_loss:.4f}")

    # Validierung nach jeder Epoche ohne Gradientenberechnung
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validation_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    val_loss /= total
    val_accuracy = 100 * correct / total

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    # Aktuellen Trainingsstand einschließlich Optimiererzustand speichern
    checkpoint_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_accuracy": val_accuracy,
        },
        checkpoint_dir / "resnet18_latest.pth"
    )

    # Bestes Modell anhand der Validierungsgenauigkeit speichern
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            checkpoint_dir / "resnet18_best.pth"
        )

        print("New best model saved!")
