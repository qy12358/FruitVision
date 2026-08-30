"""Train the binary mango/non-mango identity gate.

Positive examples come from the prepared Harumanis dataset.  Negative
examples are sampled from Fruit-360's other object classes.  This model is
intentionally separate from ripeness: a four-class ripeness softmax cannot
reject a non-mango image because it must always choose one of its four
classes.

Run from the project root:

    python training/train_mango_identifier.py
"""

import random
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.preprocessing import ImagePreprocessor

POSITIVE_DIR = (
    PROJECT_ROOT
    / "dataset"
    / "mango_harumanis"
    / "harumanis_phases_V2"
    / "images"
)
NEGATIVE_ROOTS = [
    PROJECT_ROOT / "dataset" / "fruits360-original" / "Training",
    PROJECT_ROOT / "dataset" / "fruits360-original" / "Test",
]
MULTI_SCENE_ROOT = (
    PROJECT_ROOT
    / "dataset"
    / "fruits360-multi"
    / "test-multiple_fruits"
)
MODEL_PATH = PROJECT_ROOT / "models" / "mango_identifier.keras"

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NEGATIVES_PER_CLASS = 8
HARD_NEGATIVE_PER_CLASS = 32
MULTI_SCENE_NEGATIVE_LIMIT = 256
HARD_NEGATIVE_TERMS = (
    "orange",
    "lemon",
    "lime",
    "apple",
    "peach",
    "nectarine",
    "papaya",
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(image: np.ndarray, target_size=IMAGE_SIZE) -> np.ndarray:
    """Resize without stretching the object."""
    target_w, target_h = target_size
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_size = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, new_size, interpolation=interpolation)
    canvas = np.zeros((target_h, target_w, 3), dtype=image.dtype)
    x = (target_w - new_size[0]) // 2
    y = (target_h - new_size[1]) // 2
    canvas[y:y + new_size[1], x:x + new_size[0]] = resized
    return canvas


def collect_examples() -> list[tuple[str, int]]:
    examples = []
    for path in POSITIVE_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            examples.append((str(path), 1))

    rng = random.Random(SEED)
    for root in NEGATIVE_ROOTS:
        if not root.exists():
            continue
        for class_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            paths = [
                path for path in sorted(class_dir.rglob("*"))
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
            rng.shuffle(paths)
            class_name = class_dir.name.lower()
            limit = (
                HARD_NEGATIVE_PER_CLASS
                if any(term in class_name for term in HARD_NEGATIVE_TERMS)
                else NEGATIVES_PER_CLASS
            )
            examples.extend((str(path), 0) for path in paths[:limit])

    # The multi-fruit branch has no bounding-box annotations.  Therefore it
    # is not used as a positive object-classification set: a filename such as
    # ``banana_mango.jpg`` tells us that a mango is somewhere in the scene,
    # not which pixels belong to it.  Images whose filename contains no
    # mango token are safe, useful hard negatives for rejecting cluttered
    # multi-fruit photographs at the scene gate.
    if MULTI_SCENE_ROOT.exists():
        multi_negative_paths = [
            path for path in sorted(MULTI_SCENE_ROOT.iterdir())
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and "mango" not in path.stem.lower()
            )
        ]
        examples.extend(
            (str(path), 0)
            for path in multi_negative_paths[:MULTI_SCENE_NEGATIVE_LIMIT]
        )
        print(
            "Added "
            f"{min(len(multi_negative_paths), MULTI_SCENE_NEGATIVE_LIMIT)} "
            "mango-free multi-fruit hard negatives."
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
        splits["validation"].extend(
            items[test_count:test_count + validation_count]
        )
        splits["train"].extend(items[test_count + validation_count:])
    for items in splits.values():
        rng.shuffle(items)
    return splits


class ImageSequence(Sequence):
    def __init__(self, examples, augment=False):
        self.examples = list(examples)
        self.augment = augment
        self.preprocessor = ImagePreprocessor(resize=IMAGE_SIZE)
        self.indices = np.arange(len(self.examples))
        self.augmenter = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.12,
            width_shift_range=0.08,
            height_shift_range=0.08,
            horizontal_flip=True,
            brightness_range=(0.85, 1.15),
            fill_mode="reflect",
        )
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.examples) / BATCH_SIZE))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def __getitem__(self, batch_index):
        batch_indices = self.indices[
            batch_index * BATCH_SIZE:(batch_index + 1) * BATCH_SIZE
        ]
        images, labels = [], []
        for index in batch_indices:
            path, label = self.examples[index]
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unable to read image: {path}")
            if self.augment:
                image = letterbox(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = self.augmenter.random_transform(image)
                image = cv2.cvtColor(
                    np.clip(image, 0, 255).astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )
            processed = self.preprocessor.preprocess(image)
            image = processed["model_segmented_rgb"]
            images.append(np.clip(image, 0, 255).astype(np.float32))
            labels.append(float(label))
        return np.asarray(images), np.asarray(labels, dtype=np.float32)


def build_model():
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False
    inputs = layers.Input(shape=(*IMAGE_SIZE, 3), name="image_input")
    x = base(inputs, training=False)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="mango_probability")(x)
    model = tf.keras.Model(inputs, outputs, name="MangoIdentityEfficientNetB0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model, base


def train():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    examples = collect_examples()
    splits = split_examples(examples)
    print({name: len(items) for name, items in splits.items()})

    train_sequence = ImageSequence(splits["train"], augment=True)
    validation_sequence = ImageSequence(splits["validation"], augment=False)
    train_labels = np.asarray([label for _, label in splits["train"]])
    train_counts = np.bincount(train_labels, minlength=2)
    total_train = max(1, len(train_labels))
    class_weight = {
        index: total_train / (2.0 * max(1, count))
        for index, count in enumerate(train_counts)
    }
    model, base = build_model()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
        ),
    ]
    model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=15,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Fine-tune only the final EfficientNet layers; the dataset is too small
    # to safely update the entire ImageNet backbone.
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(
        train_sequence,
        validation_data=validation_sequence,
        epochs=10,
        class_weight=class_weight,
        callbacks=callbacks,
    )
    test_metrics = model.evaluate(
        ImageSequence(splits["test"], augment=False),
        verbose=0,
        return_dict=True,
    )
    model.save(MODEL_PATH)
    print(f"Test metrics: {test_metrics}")
    print(f"Saved mango identity model to {MODEL_PATH}")


if __name__ == "__main__":
    train()
