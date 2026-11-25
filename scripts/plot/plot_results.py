
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


models = [
    "ResNet18\nLinear",
    "ViT-B/16\nLinear",
    "ViT LoRA\nr=4",
    "ViT LoRA\nr=8"
]

test_accuracy = [59.27, 63.18, 69.99, 70.77]

fig, ax = plt.subplots(figsize=(7, 4))

bars = ax.bar(models, test_accuracy)

ax.set_ylabel("Test Accuracy (%)")
ax.set_ylim(0, 80)
ax.set_title("Document Classification Performance")

for bar, value in zip(bars, test_accuracy):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

plt.savefig(FIGURES_DIR / "model_comparison.pdf", bbox_inches="tight")
plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=300, bbox_inches="tight")

plt.show()
