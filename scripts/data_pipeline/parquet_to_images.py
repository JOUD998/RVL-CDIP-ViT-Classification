
from pathlib import Path
from collections import Counter
from io import BytesIO
from PIL import Image, UnidentifiedImageError
import pyarrow.parquet as pq


# Projektverzeichnisse für Rohdaten und extrahierte Bilder
project_dir = Path(__file__).resolve().parents[2]
raw_dir = project_dir / "data" / "raw"
processed_dir = project_dir / "data" / "processed"

# Klassennamen entsprechend den ursprünglichen Label-IDs (0–15)
class_names = [
    "advertisement",
    "budget",
    "email",
    "file-folder",
    "form",
    "handwritten",
    "invoice",
    "letter",
    "memo",
    "news-article",
    "presentation",
    "questionnaire",
    "resume",
    "scientific-publication",
    "scientific-report",
    "specification",
]

splits = ["train", "validation", "test"]

total_saved = 0
total_existing = 0

# Trainings-, Validierungs- und Testdaten getrennt verarbeiten
for split in splits:
    split_dir = raw_dir / split

    print(f"\n{'=' * 50}")
    print(f"Processing split: {split.upper()}")
    print(f"{'=' * 50}")

    split_counts = Counter()

    # Alle Parquet-Dateien des jeweiligen Datensatzteils durchlaufen
    for file_path in sorted(split_dir.glob("*.parquet")):
        print(f"\nReading: {file_path.name}")

        parquet = pq.ParquetFile(file_path)
        image_index = 0
        file_saved = 0

        # Daten blockweise einlesen, um den Speicherverbrauch zu begrenzen
        for batch in parquet.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                label = int(row["label"])

                if not 0 <= label < len(class_names):
                    raise ValueError(f"Invalid label: {label}")

                class_name = class_names[label]
                image_bytes = row["image"]["bytes"]

                if not image_bytes:
                    raise ValueError(
                        f"Missing image bytes in {file_path.name}"
                    )

                # Bildformat prüfen und beschädigte Bilder überspringen
                try:
                    with Image.open(BytesIO(image_bytes)) as img:
                        image_format = img.format

                except (UnidentifiedImageError, OSError) as error:
                    print(
                        f"ERROR: {file_path.name}, "
                        f"Image index: {image_index + 1}, "
                        f"Label: {label}, "
                        f"Reason: {error}"
                    )
                    continue

                if image_format != "TIFF":
                    raise ValueError(
                        f"Unexpected format {image_format} "
                        f"in {file_path.name}"
                    )

                # Zielverzeichnis anhand des Datensatzteils und der Klasse erstellen
                output_dir = processed_dir / split / class_name
                output_dir.mkdir(parents=True, exist_ok=True)

                # Reproduzierbaren Dateinamen aus Quelldatei und Bildindex erzeugen
                image_index += 1
                output_name = (
                    f"{file_path.stem}-{image_index:05d}.tiff"
                )
                output_path = output_dir / output_name

                # Vorhandene Dateien nur akzeptieren, wenn ihr Inhalt identisch ist
                if output_path.exists():
                    if output_path.read_bytes() != image_bytes:
                        raise FileExistsError(
                            f"Existing file differs: {output_path}"
                        )
                    total_existing += 1
                else:
                    output_path.write_bytes(image_bytes)
                    file_saved += 1
                    total_saved += 1

                split_counts[label] += 1

        print(
            f"Processed: {image_index} images | "
            f"Newly saved: {file_saved}"
        )

    # Anzahl der erfolgreich verarbeiteten Bilder pro Klasse ausgeben
    print(f"\n{split.upper()} SUMMARY")
    print(f"Total images: {sum(split_counts.values())}")

    for label, count in sorted(split_counts.items()):
        print(f"  {class_names[label]}: {count}")

# Gesamtübersicht über neu gespeicherte und bereits vorhandene Bilder
print(f"\n{'=' * 50}")
print("EXTRACTION COMPLETE")
print(f"New images saved: {total_saved}")
print(f"Already existing: {total_existing}")
print(f"Total processed: {total_saved + total_existing}")
print(f"{'=' * 50}")
