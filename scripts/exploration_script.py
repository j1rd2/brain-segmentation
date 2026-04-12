import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tifffile as tiff
import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

DATASET_DIR = BASE_DIR / "data/images"
CSV_PATH = BASE_DIR / "data/data.csv"
OUTPUT_DIR = BASE_DIR / "data/exploration_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("DATASET_DIR:", DATASET_DIR)
print("CSV_PATH:", CSV_PATH)
print("DATASET_DIR exists:", DATASET_DIR.exists())
print("CSV_PATH exists:", CSV_PATH.exists())


# =========================================================
# HELPERS
# =========================================================
def is_mask_file(file_path: Path) -> bool:
    return file_path.stem.endswith("_mask")


def get_patient_id_from_filename(file_path: Path) -> str:
    """
    Example:
    TCGA_CS_4941_001.tif -> TCGA_CS_4941
    TCGA_CS_4941_001_mask.tif -> TCGA_CS_4941
    """
    stem = file_path.stem.replace("_mask", "")
    parts = stem.split("_")
    return "_".join(parts[:3])


def safe_read_tif(file_path: Path):
    try:
        return tiff.imread(str(file_path))
    except Exception as e:
        print(f"Could not read {file_path}: {e}")
        return None


def sanitize_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)


# =========================================================
# IMAGE / MASK ANALYSIS
# =========================================================
def analyze_images(dataset_dir):
    dataset_dir = Path(dataset_dir)

    tif_files = sorted(dataset_dir.rglob("*.tif"))
    image_files = [f for f in tif_files if not is_mask_file(f)]
    mask_files = [f for f in tif_files if is_mask_file(f)]

    print("=" * 60)
    print("IMAGE / MASK ANALYSIS")
    print("=" * 60)

    print(f"Total .tif files found: {len(tif_files)}")
    print(f"Number of image files: {len(image_files)}")
    print(f"Number of mask files: {len(mask_files)}")

    image_shapes = Counter()
    mask_shapes = Counter()

    pixel_class_counter = Counter()
    image_level_class_counter = Counter()
    classes_found = set()

    avg_accumulator = None
    avg_count = 0
    reference_shape = None

    images_per_patient = Counter()
    masks_per_patient = Counter()

    # -----------------------------------------------------
    # Analyze image files
    # -----------------------------------------------------
    for img_path in image_files:
        img = safe_read_tif(img_path)
        if img is None:
            continue

        image_shapes[img.shape] += 1

        patient_id = get_patient_id_from_filename(img_path)
        images_per_patient[patient_id] += 1

        # Only average images that match the first shape found
        if reference_shape is None:
            reference_shape = img.shape
            avg_accumulator = np.zeros(reference_shape, dtype=np.float64)

        if img.shape == reference_shape:
            avg_accumulator += img.astype(np.float64)
            avg_count += 1

    # -----------------------------------------------------
    # Analyze mask files
    # -----------------------------------------------------
    for mask_path in mask_files:
        mask = safe_read_tif(mask_path)
        if mask is None:
            continue

        # Convert from 0/255 to 0/1
        mask = (mask > 0).astype(np.uint8)

        mask_shapes[mask.shape] += 1

        patient_id = get_patient_id_from_filename(mask_path)
        masks_per_patient[patient_id] += 1

        unique_vals, counts = np.unique(mask, return_counts=True)
        for val, count in zip(unique_vals, counts):
            pixel_class_counter[int(val)] += int(count)
            classes_found.add(int(val))

        if np.any(mask > 0):
            image_level_class_counter["positive_mask"] += 1
        else:
            image_level_class_counter["empty_mask"] += 1

    # -----------------------------------------------------
    # Average image
    # -----------------------------------------------------
    average_image = None
    if avg_count > 0:
        average_image = avg_accumulator / avg_count

    total_mask_pixels = sum(pixel_class_counter.values())
    pixel_class_distribution = {
        cls: {
            "count": cnt,
            "percentage": (cnt / total_mask_pixels) * 100 if total_mask_pixels > 0 else 0
        }
        for cls, cnt in pixel_class_counter.items()
    }

    print("\n--- Shapes ---")
    print("Image shapes:")
    for shape, count in image_shapes.items():
        print(f"  {shape}: {count}")

    print("Mask shapes:")
    for shape, count in mask_shapes.items():
        print(f"  {shape}: {count}")

    print("\n--- Classes (from masks) ---")
    print(f"Classes found: {sorted(classes_found)}")

    print("\n--- Pixel class distribution ---")
    for cls, info in pixel_class_distribution.items():
        print(f"  Class {cls}: {info['count']} pixels ({info['percentage']:.2f}%)")

    print("\n--- Image-level class distribution ---")
    for k, v in image_level_class_counter.items():
        print(f"  {k}: {v}")

    print("\n--- Images per patient (first 10) ---")
    for patient, count in list(images_per_patient.items())[:10]:
        print(f"  {patient}: {count}")

    # -----------------------------------------------------
    # Save images per patient
    # -----------------------------------------------------
    img_per_patient_df = pd.DataFrame({
        "Patient": list(images_per_patient.keys()),
        "num_images": list(images_per_patient.values()),
        "num_masks": [masks_per_patient.get(pid, 0) for pid in images_per_patient.keys()]
    }).sort_values("Patient")

    img_per_patient_df.to_csv(OUTPUT_DIR / "images_per_patient.csv", index=False)

    # -----------------------------------------------------
    # Save shape counts
    # -----------------------------------------------------
    pd.DataFrame(
        [{"shape": str(shape), "count": count} for shape, count in image_shapes.items()]
    ).to_csv(OUTPUT_DIR / "image_shapes.csv", index=False)

    pd.DataFrame(
        [{"shape": str(shape), "count": count} for shape, count in mask_shapes.items()]
    ).to_csv(OUTPUT_DIR / "mask_shapes.csv", index=False)

    # -----------------------------------------------------
    # Save class distribution
    # -----------------------------------------------------
    pixel_dist_df = pd.DataFrame([
        {"class": cls, "pixel_count": info["count"], "percentage": info["percentage"]}
        for cls, info in pixel_class_distribution.items()
    ])
    pixel_dist_df.to_csv(OUTPUT_DIR / "mask_pixel_class_distribution.csv", index=False)

    # -----------------------------------------------------
    # Plot pixel class distribution
    # -----------------------------------------------------
    if len(pixel_class_counter) > 0:
        labels = [str(k) for k in pixel_class_counter.keys()]
        values = list(pixel_class_counter.values())
        x = np.arange(len(labels))

        plt.figure(figsize=(6, 4))
        plt.bar(x, values)
        plt.title("Mask Pixel Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Pixel count")
        plt.xticks(x, labels)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "mask_pixel_class_distribution.png", dpi=200)
        plt.close()

    # -----------------------------------------------------
    # Plot average image
    # -----------------------------------------------------
    if average_image is not None:
        np.save(OUTPUT_DIR / "average_image.npy", average_image)

        plt.figure(figsize=(12, 4))

        if average_image.ndim == 3 and average_image.shape[-1] == 3:
            for c in range(3):
                plt.subplot(1, 3, c + 1)
                plt.imshow(average_image[:, :, c], cmap="gray")
                plt.title(f"Average image - channel {c}")
                plt.axis("off")
        else:
            plt.imshow(average_image, cmap="gray")
            plt.title("Average image")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "average_image.png", dpi=200)
        plt.close()

    # -----------------------------------------------------
    # Summary txt
    # -----------------------------------------------------
    with open(OUTPUT_DIR / "image_summary.txt", "w", encoding="utf-8") as f:
        f.write("IMAGE / MASK ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total tif files: {len(tif_files)}\n")
        f.write(f"Number of images: {len(image_files)}\n")
        f.write(f"Number of masks: {len(mask_files)}\n\n")

        f.write("Image shapes:\n")
        for shape, count in image_shapes.items():
            f.write(f"  {shape}: {count}\n")

        f.write("\nMask shapes:\n")
        for shape, count in mask_shapes.items():
            f.write(f"  {shape}: {count}\n")

        f.write(f"\nMask classes: {sorted(classes_found)}\n")

        f.write("\nPixel class distribution:\n")
        for cls, info in pixel_class_distribution.items():
            f.write(f"  Class {cls}: {info['count']} pixels ({info['percentage']:.2f}%)\n")

        f.write("\nImage-level class distribution:\n")
        for k, v in image_level_class_counter.items():
            f.write(f"  {k}: {v}\n")

    return {
        "num_images": len(image_files),
        "num_masks": len(mask_files),
        "image_shapes": image_shapes,
        "mask_shapes": mask_shapes,
        "classes": sorted(classes_found),
        "pixel_class_distribution": pixel_class_distribution,
        "image_level_distribution": dict(image_level_class_counter),
        "images_per_patient_df": img_per_patient_df
    }


# =========================================================
# CSV ANALYSIS
# =========================================================
def analyze_csv(csv_path, images_per_patient_df=None):
    print("\n" + "=" * 60)
    print("CSV ANALYSIS")
    print("=" * 60)

    df = pd.read_csv(csv_path)

    print(f"CSV shape: {df.shape}")
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    # -----------------------------------------------------
    # Average age
    # -----------------------------------------------------
    avg_age = None
    if "age_at_initial_pathologic" in df.columns:
        avg_age = df["age_at_initial_pathologic"].mean()
        print(f"\nAverage age: {avg_age:.2f}")

    # -----------------------------------------------------
    # Categorical / class-like columns
    # -----------------------------------------------------
    categorical_like_cols = []

    for col in df.columns:
        if col == "Patient":
            continue

        if df[col].dtype == "object":
            categorical_like_cols.append(col)
        else:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 10:
                categorical_like_cols.append(col)

    print("\nCategorical / class-like columns:")
    for col in categorical_like_cols:
        print(f"  - {col}")

    class_distribution_results = {}

    for col in categorical_like_cols:
        try:
            counts = df[col].value_counts(dropna=False).sort_index()
        except TypeError:
            counts = df[col].value_counts(dropna=False)

        class_distribution_results[col] = counts

        print(f"\nColumn: {col}")
        print(counts)

        counts_df = counts.reset_index()
        counts_df.columns = [col, "count"]
        counts_df[col] = counts_df[col].astype("object")
        counts_df[col] = counts_df[col].where(pd.notna(counts_df[col]), "NaN")
        counts_df[col] = counts_df[col].astype(str)

        safe_col = sanitize_filename(col)
        counts_df.to_csv(OUTPUT_DIR / f"class_distribution_{safe_col}.csv", index=False)

        labels = counts_df[col].tolist()
        values = counts_df["count"].tolist()
        x = np.arange(len(labels))

        plt.figure(figsize=(8, 4))
        plt.bar(x, values)
        plt.title(f"Class distribution - {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"class_distribution_{safe_col}.png", dpi=200)
        plt.close()

    # -----------------------------------------------------
    # Correlation matrix
    # -----------------------------------------------------
    numeric_df = df.select_dtypes(include=[np.number])

    corr_matrix = numeric_df.corr()
    corr_matrix.to_csv(OUTPUT_DIR / "correlation_matrix.csv")

    if not corr_matrix.empty:
        plt.figure(figsize=(12, 8))
        plt.imshow(corr_matrix, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
        plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=200)
        plt.close()

    # -----------------------------------------------------
    # Merge with images per patient
    # -----------------------------------------------------
    merged_df = None
    if images_per_patient_df is not None and "Patient" in df.columns:
        merged_df = df.merge(images_per_patient_df, on="Patient", how="left")
        merged_df.to_csv(OUTPUT_DIR / "csv_with_images_per_patient.csv", index=False)

        print("\nImages per patient merged with CSV:")
        print(merged_df[["Patient", "num_images", "num_masks"]].head())

    # -----------------------------------------------------
    # Summary txt
    # -----------------------------------------------------
    with open(OUTPUT_DIR / "csv_summary.txt", "w", encoding="utf-8") as f:
        f.write("CSV ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"CSV shape: {df.shape}\n\n")
        f.write("Columns:\n")
        for col in df.columns:
            f.write(f"  - {col}\n")

        if avg_age is not None:
            f.write(f"\nAverage age: {avg_age:.2f}\n")

        f.write("\nCategorical / class-like columns:\n")
        for col in categorical_like_cols:
            f.write(f"  - {col}\n")

    return {
        "df": df,
        "average_age": avg_age,
        "categorical_columns": categorical_like_cols,
        "class_distributions": class_distribution_results,
        "correlation_matrix": corr_matrix,
        "merged_df": merged_df
    }


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Image files found:", len(list(DATASET_DIR.rglob("*.tif"))) if DATASET_DIR.exists() else "dir not found")

    image_results = analyze_images(DATASET_DIR)
    csv_results = analyze_csv(CSV_PATH, image_results["images_per_patient_df"])

    print("\nDone. Results saved in:", OUTPUT_DIR)