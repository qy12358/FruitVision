from pathlib import Path

from ultralytics import YOLO


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = (
    PROJECT_ROOT
    / "dataset"
    / "mango_yolo"
    / "data.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "mango_detection"
)


# ---------------------------------------------------------
# Train YOLO mango detector
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("MANGO YOLO DETECTOR TRAINING")
    print("=" * 70)

    print(f"\nDataset YAML:\n{DATA_YAML}")

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"\nCannot find data.yaml:\n{DATA_YAML}"
        )

    # Start from pretrained YOLO26 Nano.
    # Ultralytics recommends pretrained weights for custom training.
    model = YOLO("yolo26n.pt")

    results = model.train(
        data=str(DATA_YAML),

        # Training
        epochs=100,
        imgsz=640,

        # Let Ultralytics choose a suitable batch size automatically.
        batch=-1,

        # Stop if validation does not improve for 20 epochs.
        patience=20,

        # Output folder
        project=str(RUNS_DIR),
        name="mango_yolo",

        # Overwrite same experiment folder if run again.
        exist_ok=True,

        # Helpful for Windows
        workers=4,

        # Mild augmentation suitable for mango detection.
        degrees=15.0,
        translate=0.10,
        scale=0.30,
        fliplr=0.5,

        hsv_h=0.015,
        hsv_s=0.40,
        hsv_v=0.30,

        # Reduce heavy augmentation near end of training.
        close_mosaic=10,

        # Save checkpoints
        save=True,

        verbose=True,
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nYour best trained YOLO model should be here:\n"
        f"{RUNS_DIR / 'mango_yolo' / 'weights' / 'best.pt'}"
    )

    print(
        "\nYour last training checkpoint should be here:\n"
        f"{RUNS_DIR / 'mango_yolo' / 'weights' / 'last.pt'}"
    )

    return results


if __name__ == "__main__":
    main()