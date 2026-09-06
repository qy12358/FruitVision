from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = (
    PROJECT_ROOT
    / "dataset"
    / "mango_defect_seg"
    / "data.yaml"
)

RUNS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "blemish_segmentation"
)


def main():

    print("=" * 70)
    print("MANGO BLEMISH & DAMAGE SEGMENTATION TRAINING")
    print("=" * 70)

    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {DATA_YAML}"
        )

    # Pretrained segmentation model.
    model = YOLO(
        "yolo26n-seg.pt"
    )

    model.train(
        data=str(DATA_YAML),

        epochs=100,

        imgsz=640,

        batch=-1,

        patience=20,

        project=str(RUNS_DIR),

        name="mango_defect_seg",

        exist_ok=True,

        workers=4,

        # Small rotations.
        degrees=10.0,

        # Small movement.
        translate=0.08,

        # Scale augmentation.
        scale=0.25,

        # Mango may appear flipped.
        fliplr=0.5,

        # Colour/lighting augmentation.
        hsv_h=0.01,
        hsv_s=0.25,
        hsv_v=0.20,

        # Disable mosaic near training end.
        close_mosaic=10,

        save=True,

        verbose=True,
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "\nBest model:"
    )

    print(
        RUNS_DIR
        / "mango_defect_seg"
        / "weights"
        / "best.pt"
    )


if __name__ == "__main__":
    main()