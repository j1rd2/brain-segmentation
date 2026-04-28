Jesus Ramirez Delgado

# Brain MRI Segmentation for Lower-Grade Glioma

For a more in depth report take a look to the formal report: [Click here](./report.pdf).

Note: Due the high computational cost of nnU-Net, this model was trained using Google Colab GPU acceleration.

### Abastract

This project presents the development and evaluation of lightweight deep learning models for brain MRI
segmentation using the LGG Segmentation Dataset. The task focuses on binary segmentation of FLAIR
abnormality regions from 2D MRI slices with a strong class imbalance, where lesion pixels represent only
1.03% of the dataset. Several approaches were tested, starting with a threshold-based baseline and
progressing through multiple encoder-decoder architectures inspired by state-of-the-art segmentation
models. The models were evaluated using 5-fold cross-validation and a separate final test set. The final
model, encoder-decoder v4, combines skip connections, separable convolutions, MaxPooling, BCE +
Dice Loss, and a deeper but lightweight encoder-decoder structure. The results show that V4 achieved a
strong balance between segmentation performance and computational efficiency, reaching a global
cross-validation IoU of 0.520367 and a final test IoU of 0.600055, while reducing training time to
approximately 3 hours. These results indicate that the proposed lightweight architecture can provide
competitive segmentation performance while significantly reducing computational cost.

### Dataset

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

**Where to download:**[Click here](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)

## Project Structure

```text
project_root/
├── data/                  # Ignored. Stores the dataset, prepared splits, and generated data files
├── models/                # Model weights saved
├── results/               # All the results by model
├── scripts/               # Scripts used in the project (helpers, preprocessing, splitting, exploration, etc.)
├── src/                   # The base of the project, contain the models and the model interface.
├── README.md              # Project documentation
├── report.pdf             # Formal report of the project
├── requirements.txt       # Python dependencies
└── .gitignore             # Ignored files and folders
```

## How to setup the project

1. Create an .venv
2. Install al libraries using `requirements.txt` with:
```
pip install requirements.txt
```
3. Run the scripts if you want to train a model or run it.


## How to run the scripts (example):

```
python3 src/encoder_decoder_v4.py
```

- Interface Script
```
    python3 src/model_interface.py \
    --model models/encoder_decoder_v4\
    --image data/inference_image/TCGA_CS_5393_19990606_6.tif\
    --output results/inference \
    --threshold 0.5 \
    --overlay-channel 1 \
    --save-npy
```