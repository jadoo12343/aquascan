"""
AquaScan — EfficientNetB0 Training Script  (Person A deliverable)
==================================================================
Run this once to train the model and save weights to model/weights/.

Usage:
    python model/train.py --data_dir data/dataset --epochs 25 --batch_size 32

Dataset layout expected (after prepare_dataset.py runs):
    data/dataset/
        train/
            plastic/    metal/    glass/    cardboard/    paper/    trash/
        val/
            plastic/    metal/    glass/    cardboard/    paper/    trash/
        test/
            plastic/    metal/    glass/    cardboard/    paper/    trash/

Output:
    model/weights/efficientnetb0_aquascan.h5
    model/weights/training_history.json
"""

import argparse
import json
import pathlib
import sys

import numpy as np

# -- Imports guarded for early error messages ---------------------------------
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    )
    print(f"TensorFlow {tf.__version__} — GPU: {tf.config.list_physical_devices('GPU')}")
except ImportError:
    sys.exit("ERROR: tensorflow is not installed.\n"
             "Run: pip install tensorflow   (CPU) or tensorflow-gpu  (CUDA).")

# -- Class definitions ---------------------------------------------------------
CLASSES   = ["plastic", "metal", "glass", "cardboard", "paper", "trash"]
IMG_SIZE  = (224, 224)
WEIGHTS_DIR = pathlib.Path(__file__).parent / "weights"


# ------------------------------------------------------------------------------
# 1. DATA LOADERS
# ------------------------------------------------------------------------------
def make_datasets(data_dir: str, batch_size: int, seed: int = 42):
    """
    Build tf.data.Dataset pipelines from ImageFolder-style directories.
    Applies augmentation only to the training split.
    """
    ds_kwargs = dict(
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical",
        class_names=CLASSES,
        seed=seed,
    )

    train_ds = keras.utils.image_dataset_from_directory(
        pathlib.Path(data_dir) / "train", **ds_kwargs
    )
    val_ds = keras.utils.image_dataset_from_directory(
        pathlib.Path(data_dir) / "val", **ds_kwargs
    )
    test_ds = keras.utils.image_dataset_from_directory(
        pathlib.Path(data_dir) / "test", **ds_kwargs
    )

    # -- Data augmentation layer (training only) ------------------------------
    augment = keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.12),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.15),
    ], name="augmentation")

    def apply_preprocess(x, y):
        return preprocess_input(tf.cast(x, tf.float32)), y

    def apply_augment_preprocess(x, y):
        return preprocess_input(tf.cast(augment(x, training=True), tf.float32)), y

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.map(apply_augment_preprocess, num_parallel_calls=AUTOTUNE).cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(apply_preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.map(apply_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds


# ------------------------------------------------------------------------------
# 2. MODEL BUILDER
# ------------------------------------------------------------------------------
def build_model(num_classes: int = 6, dropout: float = 0.3) -> keras.Model:
    """
    EfficientNetB0 with ImageNet weights.
    Phase 1: base frozen ? only classification head trains.
    Phase 2: top-25 layers unfrozen ? fine-tune with tiny LR.
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMG_SIZE, 3),
        drop_connect_rate=0.2,
    )
    base.trainable = False     # Phase 1: freeze all

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs, name="AquaScan_EfficientNetB0")
    return model


# ------------------------------------------------------------------------------
# 3. TRAINING LOOP
# ------------------------------------------------------------------------------
def train(
    data_dir: str,
    epochs_phase1: int = 10,
    epochs_phase2: int = 15,
    batch_size: int = 32,
    lr_phase1: float = 1e-3,
    lr_phase2: float = 1e-5,
    seed: int = 42,
):
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weight_path = WEIGHTS_DIR / "efficientnetb0_aquascan.h5"

    print("\n[AquaScan] Loading datasets …")
    train_ds, val_ds, test_ds = make_datasets(data_dir, batch_size, seed)

    print("[AquaScan] Building model …")
    model = build_model()
    model.summary(print_fn=lambda x: print(" ", x))

    # -- Phase 1: Head-only training ------------------------------------------
    print(f"\n[AquaScan] Phase 1: head training for {epochs_phase1} epochs @ LR={lr_phase1}")
    model.compile(
        optimizer=keras.optimizers.Adam(lr_phase1),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_p1 = [
        EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(weight_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs_phase1, callbacks=callbacks_p1)

    # -- Phase 2: Fine-tuning top layers -------------------------------------
    print(f"\n[AquaScan] Phase 2: fine-tuning top 25 conv layers for {epochs_phase2} epochs @ LR={lr_phase2}")
    base_model = model.layers[1]          # EfficientNetB0 sub-model
    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(lr_phase2),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    callbacks_p2 = [
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(weight_path), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
    ]
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs_phase2,
        callbacks=callbacks_p2,
    )

    # -- Save training history ------------------------------------------------
    history = {}
    for k, v in hist1.history.items():
        history[f"phase1_{k}"] = [float(x) for x in v]
    for k, v in hist2.history.items():
        history[f"phase2_{k}"] = [float(x) for x in v]

    hist_path = WEIGHTS_DIR / "training_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[AquaScan] History saved ? {hist_path}")

    # -- Test set evaluation --------------------------------------------------
    print("\n[AquaScan] Evaluating on test set …")
    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"[AquaScan] Test Accuracy: {acc:.4f}   Test Loss: {loss:.4f}")
    print(f"[AquaScan] Weights saved ? {weight_path}")

    return model


# ------------------------------------------------------------------------------
# 4. ENTRY POINT
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AquaScan EfficientNetB0")
    parser.add_argument("--data_dir",     default="data/dataset", help="Root dataset directory")
    parser.add_argument("--epochs",       type=int,   default=25, help="Total epochs (split 10+15)")
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr_phase1",    type=float, default=1e-3)
    parser.add_argument("--lr_phase2",    type=float, default=1e-5)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    ep1 = max(1, args.epochs // 2)
    ep2 = args.epochs - ep1

    train(
        data_dir=args.data_dir,
        epochs_phase1=ep1,
        epochs_phase2=ep2,
        batch_size=args.batch_size,
        lr_phase1=args.lr_phase1,
        lr_phase2=args.lr_phase2,
        seed=args.seed,
    )
