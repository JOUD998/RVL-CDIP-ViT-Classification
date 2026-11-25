
from pathlib import Path
from torchvision.models import ResNet18_Weights
from PIL import ImageOps, Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader


project_dir = Path(__file__).resolve().parents[2]

train_dir = project_dir / "data" / "processed" / "train"
validation_dir = project_dir / "data" / "processed" / "validation"
test_dir = project_dir / "data" / "processed" / "test"

weights = ResNet18_Weights.DEFAULT
preprocessing = weights.transforms()

# Einheitliche Vorverarbeitung für alle Datensatzteile
transform = transforms.Compose([
    transforms.Lambda(
        lambda img: ImageOps.pad(
            img.convert("RGB"),
            (224, 224),
            method=Image.Resampling.BICUBIC,
            color="white"
        )
    ),

    # Pixelwerte von [0, 255] auf [0, 1] skalieren
    transforms.ToTensor(),

    # Normalisierung mit den Mittelwerten und Standardabweichungen der vortrainierten Gewichte
    # Formel: x_normalisiert = (x - Mittelwert) / Standardabweichung
    transforms.Normalize(
        mean=preprocessing.mean,
        std=preprocessing.std
    )


])

# ImageFolder ordnet die Bilder anhand der Verzeichnisnamen ihren Klassen zu.
# Die Bilder werden erst beim Zugriff geladen und transformiert.
train_dataset = ImageFolder(
    root=train_dir,
    transform=transform
)

validation_dataset = ImageFolder(
    root=validation_dir,
    transform=transform
)

test_dataset = ImageFolder(
    root=test_dir,
    transform=transform
)

# Trainingsdaten mischen, Validierungs- und Testdaten in fester Reihenfolge laden
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    dataset=validation_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# Kontrollausgaben zur Überprüfung der Klassen, Datenmengen und Tensorwerte
if __name__ == "__main__":
    print(train_dataset.class_to_idx == validation_dataset.class_to_idx)
    print(train_dataset.class_to_idx == test_dataset.class_to_idx)

    print("Train:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print("Test:", len(test_dataset))

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Labels:", labels)

    print("Min:", images.min().item())
    print("Max:", images.max().item())
