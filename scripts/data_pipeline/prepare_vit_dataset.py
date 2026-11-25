
from pathlib import Path
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ViT_B_16_Weights
from torch.utils.data import DataLoader

project_dir = Path(__file__).resolve().parents[2]

train_dir = project_dir / "data" / "processed" / "train"
validation_dir = project_dir / "data" / "processed" / "validation"
test_dir = project_dir / "data" / "processed" / "test"

# Vorverarbeitung anhand der vortrainierten ViT-B/16-Gewichte konfigurieren
weights = ViT_B_16_Weights.DEFAULT
preprocessing = weights.transforms()

# Einheitliche Bildgröße bei Beibehaltung des Seitenverhältnisses
# Fehlende Bildbereiche werden mit weißem Rand aufgefüllt
transform = transforms.Compose([
    transforms.Lambda(
        lambda img: ImageOps.pad(
            img.convert("RGB"),
            (224, 224),
            method=Image.Resampling.BICUBIC,
            color="white"
        )
    ),
    transforms.ToTensor(),
    # Normalisierung mit den Werten der vortrainierten ViT-Gewichte
    transforms.Normalize(
        mean=preprocessing.mean,
        std=preprocessing.std
    )
])

# Klassen anhand der Ordnerstruktur zuordnen und Bilder bei Bedarf laden
train_dataset = ImageFolder(train_dir, transform=transform)
validation_dataset = ImageFolder(validation_dir, transform=transform)
test_dataset = ImageFolder(test_dir, transform=transform)

# Kleine Batchgröße zur Begrenzung des Speicherverbrauchs
# Nur die Trainingsdaten werden bei jedem Durchlauf gemischt
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0
)

# Datenmengen, Klassenzuordnung und Tensorformen kontrollieren
if __name__ == "__main__":
    print("Train:", len(train_dataset))
    print("Validation:", len(validation_dataset))
    print("Test:", len(test_dataset))

    print(
        "Classes match:",
        train_dataset.class_to_idx
        == validation_dataset.class_to_idx
        == test_dataset.class_to_idx
    )

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
