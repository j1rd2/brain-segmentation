# Brain MRI Segmentation for Lower-Grade Glioma

This project focuses on the segmentation of abnormal regions in **brain MRI images** from patients with **lower-grade glioma (LGG)**. The dataset used contains MRI slices and their corresponding manual segmentation masks of **FLAIR abnormalities**, making this a **binary medical image segmentation** problem.

The main objective of the project is to prepare the dataset, explore its characteristics, and build a segmentation pipeline capable of identifying the abnormal region in each MRI slice.

## Project Overview

The dataset contains MRI images obtained from patients with lower-grade glioma, together with manually annotated masks that indicate the abnormal region visible in the FLAIR sequence. Each image is stored in `.tif` format and has a corresponding mask file. In addition, the dataset includes clinical and genomic information for each patient in a CSV file.

The project includes:
- dataset exploration
- preprocessing and preparation of the data
- patient-level splitting strategy
- data augmentation for training
- segmentation model development and evaluation

Still modeling and evaluation fases are under work.

## Dataset

The dataset is stored locally inside the `data/` directory. This folder is ignored from version control because it contains the raw dataset and generated prepared datasets.

Main dataset characteristics:
- **110 patients**
- **3929 MRI slices**
- **3929 segmentation masks**
- MRI image shape: **256 x 256 x 3**
- Mask shape: **256 x 256**
- Binary segmentation problem:
  - `0` = background
  - `1` = abnormal region

## Project Structure

```text
project_root/
├── data/                  # Ignored. Stores the dataset, prepared splits, and generated data files
├── scripts/               # Scripts used in the project (helpers, preprocessing, splitting, exploration, etc.)
├── README.md              # Project documentation
├── report.pdf             # Formal report of the project
├── requirements.txt       # Python dependencies
└── .gitignore             # Ignored files and folders
```

# How to setup the project
