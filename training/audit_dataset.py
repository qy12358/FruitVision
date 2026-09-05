"""Audit the prepared Harumanis dataset before training.

Run:

    python training/audit_dataset.py

The audit is read-only.  It reports label/file mismatches and exact duplicate
images assigned to more than one maturity class, which must be fixed or
excluded before model evaluation.
"""

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = (
    ROOT
    / "dataset"
    / "mango_harumanis"
    / "harumanis_phases_V2"
    / "images"
)
LABEL_FILE = ROOT / "dataset" / "mango_harumanis" / "labels original 272 images.csv"
FRUIT360_ROOT = ROOT / "dataset" / "fruits360-original"
MULTI_ROOT = ROOT / "dataset" / "fruits360-multi" / "test-multiple_fruits"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalise_label(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def main():
    files = [
        path for path in IMAGE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    ]
    counts = Counter(path.parent.name for path in files)
    print("Image counts:")
    for label, count in sorted(counts.items()):
        print(f"  {label}: {count}")

    by_stem = defaultdict(list)
    for path in files:
        by_stem[path.stem.lower()].append(path)
    # ``labels original 272 images.csv`` starts with a UTF-8 BOM, so plain
    # ``encoding="utf-8"`` makes DictReader see ``\ufeffimage_name`` instead
    # of ``image_name`` and raises KeyError.  Normalize headers as well so
    # harmless spaces in a spreadsheet export do not break the audit.
    with LABEL_FILE.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames:
            reader.fieldnames = [
                field.strip().lstrip("\ufeff")
                for field in reader.fieldnames
            ]
        rows = list(reader)
    required_columns = {"image_name", "label"}
    if not required_columns.issubset(rows[0].keys() if rows else set()):
        raise ValueError(
            f"CSV must contain columns {sorted(required_columns)}; "
            f"found {reader.fieldnames}"
        )
    missing = [
        row["image_name"] for row in rows
        if row["image_name"].lower() not in by_stem
    ]
    raw_mismatches = [
        row for row in rows
        if row["image_name"].lower() in by_stem
        and row["label"].strip().lower()
        != by_stem[row["image_name"].lower()][0].parent.name.strip().lower()
    ]
    semantic_mismatches = [
        row for row in rows
        if row["image_name"].lower() in by_stem
        and normalise_label(row["label"])
        != normalise_label(by_stem[row["image_name"].lower()][0].parent.name)
    ]
    print(f"\nCSV rows: {len(rows)}; image files: {len(files)}")
    print(f"Missing CSV files: {len(missing)}")
    print(f"Raw CSV/folder spelling mismatches: {len(raw_mismatches)}")
    print(f"Semantic CSV/folder mismatches: {len(semantic_mismatches)}")
    if raw_mismatches and not semantic_mismatches:
        print("  (The raw differences are only semi_ripe vs SEMI-RIPE.)")

    hashes = defaultdict(list)
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(path)
    conflicting = [
        paths for paths in hashes.values()
        if len({normalise_label(path.parent.name) for path in paths}) > 1
    ]
    print(f"Unique exact-image hashes: {len(hashes)}")
    print(f"Conflicting duplicate groups: {len(conflicting)}")
    for paths in conflicting[:10]:
        print("  " + " | ".join(str(path.relative_to(IMAGE_ROOT)) for path in paths))

    # Dataset-role audit.  Fruit-360 original is suitable for single-object
    # non-mango negatives; its folder list is also checked because this copy
    # does not necessarily contain a Mango class.  The multi branch has no
    # bounding boxes, so its filenames are reported as a validation manifest,
    # not converted into ripeness labels.
    if FRUIT360_ROOT.exists():
        fruit360_files = [
            path for path in FRUIT360_ROOT.rglob("*")
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]
        class_dirs = {
            path.name.lower()
            for split in ("Training", "Test", "Validation")
            if (FRUIT360_ROOT / split).exists()
            for path in (FRUIT360_ROOT / split).iterdir()
            if path.is_dir()
        }
        mango_classes = sorted(
            name for name in class_dirs if "mango" in name
        )
        print("\nFruit-360 original:")
        print(f"  image files: {len(fruit360_files)}")
        print(f"  mango class folders: {mango_classes or 'none'}")

    if MULTI_ROOT.exists():
        multi_files = [
            path for path in MULTI_ROOT.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        ]
        mango_scenes = [
            path for path in multi_files if "mango" in path.stem.lower()
        ]
        print("\nFruit-360 multi-fruit:")
        print(f"  scene images: {len(multi_files)}")
        print(f"  filenames containing mango: {len(mango_scenes)}")
        print(
            "  role: multi-object detection/validation; no bounding boxes "
            "are provided."
        )


if __name__ == "__main__":
    main()
