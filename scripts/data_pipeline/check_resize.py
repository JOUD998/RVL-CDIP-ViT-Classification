
from pathlib import Path
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# Projektverzeichnis ermitteln
project_dir = Path(__file__).resolve().parents[2]

# Beispielbild aus der Kategorie "budget" auswählen
images_dir = project_dir / "data" / "processed" / "train" / "budget"
image_path = next(images_dir.glob("*.tiff"))

# Originalbild laden und in RGB konvertieren
with Image.open(image_path) as img:
    original = img.convert("RGB")

# Seitenverhältnis beibehalten und auf 224 × 224 Pixel mit weißem Rand anpassen
resized = ImageOps.pad(
    original,
    (224, 224),
    method=Image.Resampling.BICUBIC,
    color="white"
)


# Originalbild und vorverarbeitetes Bild gegenüberstellen
fig, axes = plt.subplots(1, 2, figsize=(10, 6))

axes[0].imshow(original)
axes[0].set_title(f"Original: {original.size}")
axes[0].axis("off")

axes[1].imshow(resized)
axes[1].set_title(f"Resized: {resized.size}")
axes[1].axis("off")

plt.tight_layout()
plt.show()


# Optional: Vergleichsabbildung speichern (derzeit deaktiviert)
# output_dir = project_dir / "experiments" / "image-resize"
# output_dir.mkdir(parents=True, exist_ok=True)
#
# output_path = output_dir / "resize_comparison1.png"
#
# plt.savefig(output_path, dpi=300, bbox_inches="tight")
#
# print(f"Image saved to: {output_path}")
