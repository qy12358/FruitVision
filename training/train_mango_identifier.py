"""Train the MobileNetV2 CNN mango/non-mango identity gate.

Only repository dataset images are used. External user images remain
inference-only references and are never copied into a split.
"""

import hashlib
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessing import ImagePreprocessor
from modules.mango_identifier import fast_identity_processed

POSITIVE_ROOTS = [
    PROJECT_ROOT / "dataset" / "mango_harumanis" / "harumanis_phases_V2" / "images",
    PROJECT_ROOT / "dataset" / "mango_harumanis" / "harumanis_phases_V2 augmented",
]
FRUITS360_ROOTS = [
    PROJECT_ROOT / "dataset" / "fruits360-original" / "Training",
    PROJECT_ROOT / "dataset" / "fruits360-original" / "Test",
]
MULTI_SCENE_ROOT = PROJECT_ROOT / "dataset" / "fruits360-multi" / "test-multiple_fruits"
MODEL_PATH = PROJECT_ROOT / "models" / "mango_identifier.keras"
METADATA_PATH = PROJECT_ROOT / "models" / "mango_identifier_metadata.json"

SEED = 42
IMAGE_SIZE = (224, 224)
# Keep more examples for visually confusing non-mango fruit. In particular,
# papaya is an important hard negative because its green/oval appearance can
# otherwise be mistaken for mango by a binary gate.
MAX_NEGATIVES_PER_CLASS = 64
HARD_NEGATIVE_LIMITS = {
    "papaya": 256,
    "papaya 2": 256,
    "cactus fruit green 1": 128,
    "cactus fruit red 1": 128,
}
MULTI_SCENE_NEGATIVE_LIMIT = 512
OPERATING_THRESHOLD = 0.85
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_examples() -> list[tuple[str, int]]:
    """Collect deduplicated mango positives and many hard negatives."""
    examples = []
    positive_by_hash = {}
    for root in POSITIVE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                digest = hashlib.sha1(path.read_bytes()).hexdigest()
                positive_by_hash.setdefault(digest, path.resolve())
    examples.extend((str(path), 1) for path in sorted(positive_by_hash.values()))
    rng = random.Random(SEED)
    negative_classes = {}
    for root in FRUITS360_ROOTS:
        if not root.exists():
            continue
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if "mango" in class_dir.name.lower():
                continue
            key = class_dir.name.casefold()
            negative_classes.setdefault(key, []).extend(
                path for path in class_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )

    negative_paths = []
    for paths in negative_classes.values():
        rng.shuffle(paths)
        limit = HARD_NEGATIVE_LIMITS.get(key, MAX_NEGATIVES_PER_CLASS)
        negative_paths.extend(paths[:limit])

    if MULTI_SCENE_ROOT.exists():
        hard_scene = [
            path for path in sorted(MULTI_SCENE_ROOT.iterdir())
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
            and "mango" not in path.stem.lower()
        ]
        rng.shuffle(hard_scene)
        negative_paths.extend(hard_scene[:MULTI_SCENE_NEGATIVE_LIMIT])

    examples.extend((str(path), 0) for path in negative_paths)
    negative_count = len(negative_paths)

    print(
        f"Identity sources: {len([x for x in examples if x[1] == 1])} mango positives, "
        f"{negative_count} non-mango negatives."
    )
    if not examples:
        raise RuntimeError("No training images were found for the mango gate.")
    return examples


def split_examples(examples):
    rng = random.Random(SEED)
    by_label = {0: [], 1: []}
    for item in examples:
        by_label[item[1]].append(item)
    splits = {"train": [], "validation": [], "test": []}
    for label, items in by_label.items():
        rng.shuffle(items)
        test_count = max(1, int(round(len(items) * 0.15)))
        validation_count = max(1, int(round(len(items) * 0.15)))
        splits["test"].extend(items[:test_count])
        splits["validation"].extend(items[test_count:test_count + validation_count])
        splits["train"].extend(items[test_count + validation_count:])
    for items in splits.values():
        rng.shuffle(items)
    return splits


def prepare_images(examples, preprocessor):
    """Return segmented and raw views matching application inference."""
    images, labels = [], []
    for index, (path, label) in enumerate(examples, start=1):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable training image: {path}")
        # Use the lightweight crop/mask path offline. It has the same
        # foreground-centered behavior as the CNN input at inference, but
        # avoids running GrabCut thousands of times during a training run.
        processed = fast_identity_processed(image, IMAGE_SIZE)
        segmented = processed["model_segmented_rgb"]
        raw = cv2.cvtColor(preprocessor.resize_image(image), cv2.COLOR_BGR2RGB)
        images.extend((segmented, raw))
        labels.extend((label, label))
        if index % 250 == 0:
            print(f"Prepared {index}/{len(examples)} source images.")
    return np.asarray(images, dtype=np.uint8), np.asarray(labels, dtype=np.float32)


def build_cnn(tf):
    candidate_path = MODEL_PATH.with_name("mango_identifier_cnn_candidate.keras")
    if candidate_path.exists():
        return tf.keras.models.load_model(candidate_path, compile=False)
    if MODEL_PATH.exists():
        return tf.keras.models.load_model(MODEL_PATH, compile=False)

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE[::-1], 3), name="image_input")
    try:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(*IMAGE_SIZE[::-1], 3)
        )
    except Exception:
        backbone = tf.keras.applications.MobileNetV2(
            include_top=False, weights=None, input_shape=(*IMAGE_SIZE[::-1], 3)
        )
    backbone.trainable = False
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="input_scaling")(inputs)
    x = backbone(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="mango_probability")(x)
    return tf.keras.Model(inputs, outputs)


def train():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required to train the CNN identity gate.") from exc

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    splits = split_examples(collect_examples())
    print({name: len(items) for name, items in splits.items()})
    preprocessor = ImagePreprocessor(resize=IMAGE_SIZE)
    train_x, train_y = prepare_images(splits["train"], preprocessor)
    validation_x, validation_y = prepare_images(splits["validation"], preprocessor)
    test_x, test_y = prepare_images(splits["test"], preprocessor)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.06),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.12),
    ], name="identity_augmentation")

    def augment(images, labels):
        return augmentation(images, training=True), labels

    train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    train_ds = train_ds.shuffle(len(train_y), seed=SEED, reshuffle_each_iteration=True)
    train_ds = train_ds.batch(32).map(augment, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    validation_ds = tf.data.Dataset.from_tensor_slices((validation_x, validation_y)).batch(32)

    model = build_cnn(tf)
    backbone = next(
        (layer for layer in model.layers if "mobilenetv2" in layer.name.lower()),
        None,
    )
    if backbone is not None:
        backbone.trainable = False
    negative_count = max(1, int(np.sum(train_y == 0)))
    positive_count = max(1, int(np.sum(train_y == 1)))
    class_weight = {
        0: 0.5 * len(train_y) / negative_count,
        1: 0.5 * len(train_y) / positive_count,
    }
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    candidate_path = MODEL_PATH.with_name("mango_identifier_cnn_candidate.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(candidate_path, monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=1, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.3, patience=1, min_lr=1e-7),
    ]
    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=3,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    if candidate_path.exists():
        model = tf.keras.models.load_model(candidate_path, compile=False)
    model.save(MODEL_PATH)

    probabilities = model.predict(test_x, batch_size=32, verbose=0).reshape(-1)
    source_probabilities = probabilities.reshape(-1, 2).mean(axis=1)
    source_labels = test_y.reshape(-1, 2)[:, 0]
    predictions = source_probabilities >= OPERATING_THRESHOLD
    accuracy = float(np.mean(predictions == source_labels))
    tp = int(np.sum(predictions & (source_labels == 1)))
    tn = int(np.sum(~predictions & (source_labels == 0)))
    fp = int(np.sum(predictions & (source_labels == 0)))
    fn = int(np.sum(~predictions & (source_labels == 1)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Mango precision: {precision * 100:.2f}% | mango recall: {recall * 100:.2f}%")
    print(f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.write_text(json.dumps({
        "model": MODEL_PATH.name,
        "model_type": "MobileNetV2 binary CNN",
        "input_size": list(IMAGE_SIZE),
        "positive_label": "mango",
        "negative_label": "non-mango",
        "threshold": OPERATING_THRESHOLD,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "epochs_trained": len(history.history.get("loss", [])),
    }, indent=2), encoding="utf-8")
    if candidate_path.exists():
        candidate_path.unlink()
    print(f"Saved CNN identity model to {MODEL_PATH}")
    print(f"Saved training metadata to {METADATA_PATH}")


if __name__ == "__main__":
    train()
