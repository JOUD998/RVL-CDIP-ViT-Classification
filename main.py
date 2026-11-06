from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq

# Locate the file relative to main.py
project_dir = Path(__file__).resolve().parent
file_path = project_dir / "data/raw/train-00020-of-00119.parquet"

parquet = pq.ParquetFile(file_path)

print("File:", file_path.name)
print("Size (MiB):", round(file_path.stat().st_size / 1024**2, 2))
print("Rows:", parquet.metadata.num_rows)
print("\nColumn names and types:")
print(parquet.schema_arrow)

# Inspect three records without printing the image bytes
sample = next(parquet.iter_batches(batch_size=3))

print("\nSample records:")
for index, row in enumerate(sample.to_pylist()):
    image = row["image"]
    image_bytes = image["bytes"]

    print(
        f"Record {index}:",
        f"label={row['label']},",
        f"image_size={len(image_bytes) if image_bytes else 0} bytes,",
        f"path={image['path']}",
    )

# Count labels without loading image data
counts = Counter()

for batch in parquet.iter_batches(columns=["label"]):
    counts.update(batch.column(0).to_pylist())

print("\nRecords per label:")
for label, count in sorted(counts.items()):
    print(f"Label {label}: {count}")