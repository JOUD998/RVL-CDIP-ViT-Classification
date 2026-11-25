
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "results/vit_lora_r8/test/confusion_matrix.csv"
OUTPUT = ROOT / "figures/confusion_matrix_lora_r8.pdf"

# Read class names directly from CSV
cm = pd.read_csv(INPUT, index_col=0)

# Validate labels and matrix
if cm.shape[0] != cm.shape[1]:
    raise ValueError("Confusion matrix must be square.")

if list(cm.index) != list(cm.columns):
    raise ValueError(
        "True and predicted class labels do not match."
    )

cm = cm.apply(pd.to_numeric)

# Normalize by true class (row percentages)
cm_percent = cm.div(
    cm.sum(axis=1),
    axis=0
).fillna(0) * 100

# Plot
fig, ax = plt.subplots(figsize=(12, 10))

sns.heatmap(
    cm_percent,
    cmap="Blues",
    vmin=0,
    vmax=100,
    annot=False,
    square=True,
    linewidths=0.2,
    cbar_kws={"label": "Percentage (%)"},
    ax=ax
)

# Show class accuracy on diagonal
for i in range(len(cm)):
    value = cm_percent.iloc[i, i]

    ax.text(
        i + 0.5,
        i + 0.5,
        f"{value:.0f}%",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        color="white" if value >= 55 else "black"
    )

ax.set_xlabel("Predicted class", fontsize=12)
ax.set_ylabel("True class", fontsize=12)

ax.set_title(
    "ViT-B/16 with LoRA (r=8) — Test Set",
    fontsize=14
)

ax.tick_params(axis="x", labelrotation=55)
ax.tick_params(axis="y", labelrotation=0)

plt.setp(
    ax.get_xticklabels(),
    ha="right",
    rotation_mode="anchor"
)

fig.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT, bbox_inches="tight")
plt.close(fig)

# Verify overall accuracy
accuracy = (
    cm.to_numpy().trace()
    / cm.to_numpy().sum()
    * 100
)

print(f"Saved: {OUTPUT}")
print(f"Classes: {len(cm)}")
print(f"Test accuracy: {accuracy:.2f}%")
