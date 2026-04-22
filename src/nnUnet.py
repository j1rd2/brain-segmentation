import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "data" / "prepared_dataset"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

MODEL_NAME = "nnunet"
MODEL_DIR = MODELS_DIR / MODEL_NAME
MODEL_RESULTS_DIR = RESULTS_DIR / MODEL_NAME

N_FOLDS = 5
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
THRESHOLD = 0.5
SEED = 42

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MODEL_RESULTS_DIR, exist_ok=True)

tf.random.set_seed(SEED)
np.random.seed(SEED)


# =========================================================
# HELPERS
# =========================================================
def is_mask_file(path: Path) -> bool:
    return path.stem.endswith("_mask")


def normalize_image(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)

    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[-1]):
        ch = image[:, :, c]
        ch_min = ch.min()
        ch_max = ch.max()

        if ch_max - ch_min == 0:
            out[:, :, c] = 0.0
        else:
            out[:, :, c] = (ch - ch_min) / (ch_max - ch_min)

    return out


def load_image_and_mask(img_path: Path, mask_path: Path):
    image = tiff.imread(str(img_path))
    mask = tiff.imread(str(mask_path))

    image = normalize_image(image)
    mask = (mask > 0).astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)

    return image, mask


def collect_pairs(split_dir: Path):
    all_tifs = sorted(split_dir.rglob("*.tif"))
    image_files = [p for p in all_tifs if not is_mask_file(p)]

    pairs = []
    for img_path in image_files:
        mask_path = img_path.with_name(img_path.stem + "_mask.tif")
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            print(f"Warning: mask not found for {img_path}")

    return pairs


def load_dataset_from_pairs(pairs):
    images = []
    masks = []
    meta = []

    for img_path, mask_path in pairs:
        image, mask = load_image_and_mask(img_path, mask_path)
        images.append(image)
        masks.append(mask)

        meta.append({
            "patient": img_path.parent.name,
            "image_name": img_path.name,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        })

    X = np.array(images, dtype=np.float32)
    y = np.array(masks, dtype=np.float32)
    meta_df = pd.DataFrame(meta)

    return X, y, meta_df


# =========================================================
# LOSSES / METRICS
# =========================================================
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denom = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (denom + smooth)
    return tf.reduce_mean(dice)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce = tf.reduce_mean(bce)
    return bce + dice_loss(y_true, y_pred)


# =========================================================
# MODEL - nnU-Net-inspired 2D U-Net
# =========================================================
def conv_block(x, filters, name_prefix, dropout_rate=0.0):
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, name=f"{name_prefix}_conv1")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
    x = layers.LeakyReLU(alpha=0.01, name=f"{name_prefix}_lrelu1")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False, name=f"{name_prefix}_conv2")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)
    x = layers.LeakyReLU(alpha=0.01, name=f"{name_prefix}_lrelu2")(x)

    if dropout_rate > 0:
        x = layers.SpatialDropout2D(dropout_rate, name=f"{name_prefix}_dropout")(x)

    return x


def encoder_block(x, filters, name_prefix, dropout_rate=0.0):
    feat = conv_block(x, filters, name_prefix=name_prefix, dropout_rate=dropout_rate)
    down = layers.MaxPooling2D((2, 2), name=f"{name_prefix}_pool")(feat)
    return feat, down


def decoder_block(x, skip, filters, name_prefix, dropout_rate=0.0):
    x = layers.UpSampling2D((2, 2), interpolation="bilinear", name=f"{name_prefix}_upsample")(x)
    x = layers.Concatenate(name=f"{name_prefix}_concat")([x, skip])
    x = conv_block(x, filters, name_prefix=name_prefix, dropout_rate=dropout_rate)
    return x


def build_nnunet_2d_like(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape, name="input")

    # Encoder
    skip1, x = encoder_block(inputs, 32, "enc1")
    skip2, x = encoder_block(x, 64, "enc2")
    skip3, x = encoder_block(x, 128, "enc3")
    skip4, x = encoder_block(x, 256, "enc4", dropout_rate=0.10)

    # Bottleneck
    x = conv_block(x, 320, "bottleneck", dropout_rate=0.20)

    # Decoder
    x = decoder_block(x, skip4, 256, "dec4", dropout_rate=0.10)
    x = decoder_block(x, skip3, 128, "dec3")
    x = decoder_block(x, skip2, 64, "dec2")
    x = decoder_block(x, skip1, 32, "dec1")

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid", padding="same", name="output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=MODEL_NAME)
    return model


def save_model_config():
    config = {
        "model_name": MODEL_NAME,
        "framework": "Keras / TensorFlow",
        "input_shape": [256, 256, 3],
        "output_channels": 1,
        "task": "binary image segmentation",
        "architecture": {
            "type": "nnU-Net-inspired 2D U-Net",
            "encoder_block_1": ["Conv2D(32,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(32,3x3)", "BatchNorm", "LeakyReLU", "MaxPool"],
            "encoder_block_2": ["Conv2D(64,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(64,3x3)", "BatchNorm", "LeakyReLU", "MaxPool"],
            "encoder_block_3": ["Conv2D(128,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(128,3x3)", "BatchNorm", "LeakyReLU", "MaxPool"],
            "encoder_block_4": ["Conv2D(256,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(256,3x3)", "BatchNorm", "LeakyReLU", "SpatialDropout2D(0.10)", "MaxPool"],
            "bottleneck": ["Conv2D(320,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(320,3x3)", "BatchNorm", "LeakyReLU", "SpatialDropout2D(0.20)"],
            "decoder_block_4": ["UpSampling", "Concat skip", "Conv2D(256,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(256,3x3)", "BatchNorm", "LeakyReLU"],
            "decoder_block_3": ["UpSampling", "Concat skip", "Conv2D(128,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(128,3x3)", "BatchNorm", "LeakyReLU"],
            "decoder_block_2": ["UpSampling", "Concat skip", "Conv2D(64,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(64,3x3)", "BatchNorm", "LeakyReLU"],
            "decoder_block_1": ["UpSampling", "Concat skip", "Conv2D(32,3x3)", "BatchNorm", "LeakyReLU", "Conv2D(32,3x3)", "BatchNorm", "LeakyReLU"],
            "output": ["Conv2D(1,1x1)", "Sigmoid"],
        },
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "bce_dice_loss",
        "metric": "dice_coefficient",
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "prediction_threshold": THRESHOLD,
    }

    with open(MODEL_DIR / f"{MODEL_NAME}_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


# =========================================================
# EVALUATION METRICS
# =========================================================
def confusion_from_masks(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.uint8).flatten()
    y_pred = y_pred.astype(np.uint8).flatten()

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return tp, fp, tn, fn


def compute_metrics(tp: int, fp: int, tn: int, fn: int):
    eps = 1e-8

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    pixel_accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "IoU": iou,
        "Precision": precision,
        "Recall": recall,
        "Pixel Accuracy": pixel_accuracy,
    }


def evaluate_predictions(y_true, y_pred_prob, threshold=0.5, meta_df=None, split_name="val", fold_name=None):
    y_pred_bin = (y_pred_prob >= threshold).astype(np.uint8)
    y_true_bin = (y_true >= 0.5).astype(np.uint8)

    per_image_rows = []

    total_tp = total_fp = total_tn = total_fn = 0

    for i in range(len(y_true_bin)):
        tp, fp, tn, fn = confusion_from_masks(y_true_bin[i], y_pred_bin[i])
        metrics = compute_metrics(tp, fp, tn, fn)

        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn

        row = {
            "split": split_name,
            "fold": fold_name,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "IoU": metrics["IoU"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "Pixel Accuracy": metrics["Pixel Accuracy"],
        }

        if meta_df is not None:
            row["patient"] = meta_df.iloc[i]["patient"]
            row["image_name"] = meta_df.iloc[i]["image_name"]

        per_image_rows.append(row)

    summary_metrics = compute_metrics(total_tp, total_fp, total_tn, total_fn)

    summary_row = {
        "split": split_name,
        "fold": fold_name,
        "TP": total_tp,
        "FP": total_fp,
        "TN": total_tn,
        "FN": total_fn,
        "IoU": summary_metrics["IoU"],
        "Precision": summary_metrics["Precision"],
        "Recall": summary_metrics["Recall"],
        "Pixel Accuracy": summary_metrics["Pixel Accuracy"],
    }

    return pd.DataFrame(per_image_rows), summary_row


# =========================================================
# TRAIN / EVAL
# =========================================================
def train_and_evaluate_fold(fold_num: int):
    print(f"\n=== Fold {fold_num} ===")

    train_dir = DATASET_DIR / f"fold_{fold_num}" / "train"
    val_dir = DATASET_DIR / f"fold_{fold_num}" / "val"

    train_pairs = collect_pairs(train_dir)
    val_pairs = collect_pairs(val_dir)

    print(f"Train pairs: {len(train_pairs)}")
    print(f"Val pairs: {len(val_pairs)}")

    X_train, y_train, train_meta = load_dataset_from_pairs(train_pairs)
    X_val, y_val, val_meta = load_dataset_from_pairs(val_pairs)

    model = build_nnunet_2d_like(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=bce_dice_loss,
        metrics=[dice_coefficient]
    )

    model_path = MODEL_DIR / f"{MODEL_NAME}_fold_{fold_num}.h5"
    history_csv_path = MODEL_RESULTS_DIR / f"{MODEL_NAME}_fold_{fold_num}_history.csv"

    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        CSVLogger(str(history_csv_path)),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks
    )

    best_model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coefficient": dice_coefficient,
            "dice_loss": dice_loss
        }
    )

    val_pred = best_model.predict(X_val, batch_size=BATCH_SIZE, verbose=1)
    val_per_image_df, val_summary = evaluate_predictions(
        y_true=y_val,
        y_pred_prob=val_pred,
        threshold=THRESHOLD,
        meta_df=val_meta,
        split_name="val",
        fold_name=f"fold_{fold_num}"
    )

    val_per_image_df.to_csv(
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_fold_{fold_num}_val_per_image.csv",
        index=False
    )

    return best_model, val_per_image_df, val_summary, history.history


def evaluate_final_test_with_model(model, model_tag="best_fold"):
    test_dir = DATASET_DIR / "final_test"

    if not test_dir.exists():
        print("Final test directory not found. Skipping final test evaluation.")
        return None, None

    test_pairs = collect_pairs(test_dir)
    print(f"Final test pairs: {len(test_pairs)}")

    X_test, y_test, test_meta = load_dataset_from_pairs(test_pairs)
    test_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)

    test_per_image_df, test_summary = evaluate_predictions(
        y_true=y_test,
        y_pred_prob=test_pred,
        threshold=THRESHOLD,
        meta_df=test_meta,
        split_name="final_test",
        fold_name=model_tag
    )

    return test_per_image_df, test_summary


def save_text_summary(df: pd.DataFrame, output_path: Path, title: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * 100 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n")


# =========================================================
# MAIN
# =========================================================
def main():
    save_model_config()

    fold_summaries = []
    all_val_per_image = []
    best_fold_model = None
    best_fold_name = None
    best_val_iou = -1.0

    for fold_num in range(1, N_FOLDS + 1):
        model, val_per_image_df, val_summary, history_dict = train_and_evaluate_fold(fold_num)

        fold_summaries.append(val_summary)
        all_val_per_image.append(val_per_image_df)

        if val_summary["IoU"] > best_val_iou:
            best_val_iou = val_summary["IoU"]
            best_fold_model = model
            best_fold_name = f"fold_{fold_num}"

    val_summary_df = pd.DataFrame(fold_summaries)

    global_tp = int(val_summary_df["TP"].sum())
    global_fp = int(val_summary_df["FP"].sum())
    global_tn = int(val_summary_df["TN"].sum())
    global_fn = int(val_summary_df["FN"].sum())

    global_metrics = compute_metrics(global_tp, global_fp, global_tn, global_fn)

    global_row = pd.DataFrame([{
        "split": "val_global_cv",
        "fold": "global_cv",
        "TP": global_tp,
        "FP": global_fp,
        "TN": global_tn,
        "FN": global_fn,
        "IoU": global_metrics["IoU"],
        "Precision": global_metrics["Precision"],
        "Recall": global_metrics["Recall"],
        "Pixel Accuracy": global_metrics["Pixel Accuracy"],
    }])

    val_summary_with_global = pd.concat([val_summary_df, global_row], ignore_index=True)

    val_summary_with_global.to_csv(
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_cv_summary.csv",
        index=False
    )

    all_val_per_image_df = pd.concat(all_val_per_image, ignore_index=True)
    all_val_per_image_df.to_csv(
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_all_val_per_image.csv",
        index=False
    )

    if best_fold_model is not None:
        test_per_image_df, test_summary = evaluate_final_test_with_model(
            best_fold_model,
            model_tag=best_fold_name
        )

        if test_per_image_df is not None and test_summary is not None:
            test_per_image_df.to_csv(
                MODEL_RESULTS_DIR / f"{MODEL_NAME}_final_test_per_image.csv",
                index=False
            )

            test_summary_df = pd.DataFrame([test_summary])
            test_summary_df.to_csv(
                MODEL_RESULTS_DIR / f"{MODEL_NAME}_final_test_summary.csv",
                index=False
            )
        else:
            test_summary_df = pd.DataFrame()
    else:
        test_summary_df = pd.DataFrame()

    combined_summary_df = val_summary_with_global.copy()

    if not test_summary_df.empty:
        combined_summary_df = pd.concat([combined_summary_df, test_summary_df], ignore_index=True)

    combined_summary_df.to_csv(
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_summary_table.csv",
        index=False
    )

    save_text_summary(
        combined_summary_df,
        MODEL_RESULTS_DIR / f"{MODEL_NAME}_summary_table.txt",
        title=f"{MODEL_NAME.upper()} SUMMARY TABLE"
    )

    print("\n=== Final Summary Table ===")
    print(combined_summary_df.to_string(index=False))
    print(f"\nBest fold model: {best_fold_name} (Val IoU = {best_val_iou:.6f})")
    print(f"Models saved in: {MODEL_DIR}")
    print(f"Results saved in: {MODEL_RESULTS_DIR}")


if __name__ == "__main__":
    main()