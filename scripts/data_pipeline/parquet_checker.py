
import pyarrow.parquet as pq
from pathlib import Path
from collections import Counter

# Parquet-Datei mit den Validierungsdaten laden
project_dir = Path(__file__).resolve().parents[2]
file_path = project_dir / "data/raw/validation/val-specification-200.parquet"

parquet = pq.ParquetFile(file_path)

print("Total images:", parquet.metadata.num_rows)

# Klassenverteilung anhand der Labels überprüfen
labels = parquet.read(columns=["label"]).column("label")

for label, count in sorted(Counter(labels).items()):
    print(f"Label {label}: {count} images")
