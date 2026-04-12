import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import tifffile as tiff
import albumentations as A

from sklearn.model_selection import GroupShuffleSplit, GroupKFold


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "data/images"
OUTPUT_DIR = BASE_DIR / "data/prepared_dataset"

RANDOM_SEED = 42
TEST_SIZE = 0.10          # 10% final test patients
N_FOLDS = 5               # 5-fold cross validation on remaining patients
AUGS_PER_IMAGE = 2        # how many augmented samples to create per train image

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# =========================================================
# SAFE AUGMENTATION PIPELINE
# =========================================================
train_augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=0, p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.03,
        scale_limit=0.05,
        rotate_limit=0,
        border_mode=0,
        p=0.5
    ),
    A.RandomBrightnessContrast(
        brightness_limit=0.08,
        contrast_limit=0.08,
        p=0.3
    ),
])


# =========================================================
# HELPERS
# =========================================================
def is_mask_file(path: Path) -> bool:
    return path.stem.endswith("_mask")


def patient_id_from_name(path: Path) -> str:
    """
    Example:
    TCGA_CS_4941_001.tif -> TCGA_CS_4941
    TCGA_CS_4941_001_mask.tif -> TCGA_CS_4941
    """
    stem = path.stem.replace("_mask", "")
    parts = stem.split("_")
    return "_".join(parts[:3])


def build_patient_index(dataset_dir: Path):
    """
    Returns:
      patient_to_pairs = {
          patient_id: [
              (image_path, mask_path),
              ...
          ]
      }
    """
    all_tifs = sorted(dataset_dir.rglob("*.tif"))

    image_files = [p for p in all_tifs if not is_mask_file(p)]
    mask_files = [p for p in all_tifs if is_mask_file(p)]

    mask_map = {p.stem: p for p in mask_files}
    patient_to_pairs = defaultdict(list)

    for img_path in image_files:
        mask_stem = img_path.stem + "_mask"
        if mask_stem not in mask_map:
            print(f"Warning: mask not found for {img_path.name}")
            continue

        mask_path = mask_map[mask_stem]
        patient_id = patient_id_from_name(img_path)
        patient_to_pairs[patient_id].append((img_path, mask_path))

    return dict(patient_to_pairs)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def copy_pair(img_path: Path, mask_path: Path, out_dir: Path):
    ensure_dir(out_dir)
    shutil.copy2(img_path, out_dir / img_path.name)
    shutil.copy2(mask_path, out_dir / mask_path.name)


def save_augmented_pair(image, mask, original_img_name: str, out_dir: Path, aug_idx: int):
    ensure_dir(out_dir)

    stem = Path(original_img_name).stem
    img_out = out_dir / f"{stem}_aug{aug_idx:02d}.tif"
    mask_out = out_dir / f"{stem}_aug{aug_idx:02d}_mask.tif"

    # keep MRI image as uint8 if possible
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    # force binary mask 0/255 for saved dataset
    mask = (mask > 0).astype(np.uint8) * 255

    tiff.imwrite(str(img_out), image)
    tiff.imwrite(str(mask_out), mask)


def augment_and_save_pair(img_path: Path, mask_path: Path, out_dir: Path, n_augs: int):
    image = tiff.imread(str(img_path))
    mask = tiff.imread(str(mask_path))

    # binarize mask to 0/1 for augmentation
    mask = (mask > 0).astype(np.uint8)

    for i in range(1, n_augs + 1):
        augmented = train_augmentation(image=image, mask=mask)
        aug_image = augmented["image"]
        aug_mask = augmented["mask"]

        save_augmented_pair(aug_image, aug_mask, img_path.name, out_dir, i)


def summarize_split(name: str, patients: list, patient_to_pairs: dict):
    n_images = sum(len(patient_to_pairs[p]) for p in patients)
    print(f"{name}: {len(patients)} patients, {n_images} image-mask pairs")


# =========================================================
# MAIN SPLIT LOGIC
# =========================================================
def create_splits_and_dataset():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    patient_to_pairs = build_patient_index(DATASET_DIR)
    patients = sorted(patient_to_pairs.keys())

    if len(patients) == 0:
        raise ValueError("No patients found in dataset.")

    print(f"Found {len(patients)} patients.")

    # -----------------------------------------------------
    # 1) Final test split at patient level
    # -----------------------------------------------------
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    dummy_X = np.zeros(len(patients))
    groups = np.array(patients)

    trainval_idx, test_idx = next(gss.split(dummy_X, groups=groups))
    trainval_patients = [patients[i] for i in trainval_idx]
    test_patients = [patients[i] for i in test_idx]

    print("\nFinal split:")
    summarize_split("Train+Val pool", trainval_patients, patient_to_pairs)
    summarize_split("Final Test", test_patients, patient_to_pairs)

    # -----------------------------------------------------
    # 2) Save final test set once
    # -----------------------------------------------------
    test_dir = OUTPUT_DIR / "final_test"
    ensure_dir(test_dir)

    for patient_id in test_patients:
        patient_out = test_dir / patient_id
        for img_path, mask_path in patient_to_pairs[patient_id]:
            copy_pair(img_path, mask_path, patient_out)

    # -----------------------------------------------------
    # 3) 5-fold cross validation on remaining patients
    # -----------------------------------------------------
    gkf = GroupKFold(n_splits=N_FOLDS)

    trainval_dummy_X = np.zeros(len(trainval_patients))
    trainval_groups = np.array(trainval_patients)

    fold_records = []

    for fold_num, (train_idx, val_idx) in enumerate(
        gkf.split(trainval_dummy_X, groups=trainval_groups),
        start=1
    ):
        fold_dir = OUTPUT_DIR / f"fold_{fold_num}"
        train_dir = fold_dir / "train"
        val_dir = fold_dir / "val"

        train_patients = [trainval_patients[i] for i in train_idx]
        val_patients = [trainval_patients[i] for i in val_idx]

        print(f"\nFold {fold_num}")
        summarize_split("Train", train_patients, patient_to_pairs)
        summarize_split("Val", val_patients, patient_to_pairs)

        # -------------------------
        # Save validation data
        # -------------------------
        for patient_id in val_patients:
            patient_out = val_dir / patient_id
            for img_path, mask_path in patient_to_pairs[patient_id]:
                copy_pair(img_path, mask_path, patient_out)

        # -------------------------
        # Save training data
        # Original + augmented
        # -------------------------
        for patient_id in train_patients:
            patient_out = train_dir / patient_id

            for img_path, mask_path in patient_to_pairs[patient_id]:
                # Copy original pair
                copy_pair(img_path, mask_path, patient_out)

                # Create augmented versions
                augment_and_save_pair(
                    img_path=img_path,
                    mask_path=mask_path,
                    out_dir=patient_out,
                    n_augs=AUGS_PER_IMAGE
                )

        fold_records.append({
            "fold": fold_num,
            "train_patients": len(train_patients),
            "val_patients": len(val_patients),
            "train_pairs_original": sum(len(patient_to_pairs[p]) for p in train_patients),
            "val_pairs": sum(len(patient_to_pairs[p]) for p in val_patients),
        })

    # -----------------------------------------------------
    # 4) Save summary file
    # -----------------------------------------------------
    summary_path = OUTPUT_DIR / "split_summary.txt"
    ensure_dir(OUTPUT_DIR)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("DATASET SPLIT SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total patients: {len(patients)}\n")
        f.write(f"Final test patients: {len(test_patients)}\n")
        f.write(f"Train+Val patients: {len(trainval_patients)}\n")
        f.write(f"Number of folds: {N_FOLDS}\n")
        f.write(f"Augmentations per train image: {AUGS_PER_IMAGE}\n\n")

        for record in fold_records:
            f.write(
                f"Fold {record['fold']}: "
                f"train_patients={record['train_patients']}, "
                f"val_patients={record['val_patients']}, "
                f"train_pairs_original={record['train_pairs_original']}, "
                f"val_pairs={record['val_pairs']}\n"
            )

    print(f"\nDone. Prepared dataset saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    create_splits_and_dataset()