import pyarrow.parquet as pq
from pathlib import Path
from collections import Counter

project_dir = Path(__file__).resolve().parent.parent
file_path = project_dir / "data/raw/validation/val-specification-200.parquet"

parquet = pq.ParquetFile(file_path)

print("Total images:", parquet.metadata.num_rows)

labels = parquet.read(columns=["label"]).column("label").to_pylist()


for label, count in sorted(Counter(labels).items()):
    print(f"Label {label}: {count} images")