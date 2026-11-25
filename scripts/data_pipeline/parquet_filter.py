
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np

project_dir = Path(__file__).resolve().parents[2]

input_file = project_dir / "data/raw/validation/val-000142-of-00015.parquet"
output_file = project_dir / "data/raw/validation/val-specification-200.parquet"

table = pq.read_table(input_file)

# Indizes aller Bilder mit dem Label 15 ermitteln
labels = table.column("label").to_numpy()
budget_indices = np.where(labels == 15)[0]

# 200 Bilder zufällig und ohne Wiederholung auswählen
# Der feste Seed gewährleistet eine reproduzierbare Auswahl
rng = np.random.default_rng(seed=42)
selected_indices = rng.choice(budget_indices, size=200, replace=False)

# Ausgewählte Datensätze extrahieren
sampled_table = table.take(selected_indices)

# Teilmenge als komprimierte Parquet-Datei speichern
pq.write_table(sampled_table, output_file, compression="zstd")

print("Saved:", output_file.name)
print("Total images:", sampled_table.num_rows)
