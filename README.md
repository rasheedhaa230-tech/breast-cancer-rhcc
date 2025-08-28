# RHCC: Reconstructed Hough Contour Classification for Breast Cancer Detection

This repository contains the implementation for the paper "Enhanced Breast Cancer Detection Using 3D Reconstructed Mammograms with Active Contour Segmentation and Deep Learning-Based RHCC Framework".

## Overview
The proposed RHCC method aims to improve breast cancer classification by converting 2D mammograms into 3D volumes using Harmonic Fourier Transform, segmenting regions of interest with Active Contours, and classifying them with a 3D CNN. This code provides a reproducible baseline for the study, including data loading, preprocessing, and the defined baseline models ("Conventional CNN" and "Conventional DL").

## Dataset
1.  Download the CBIS-DDSM dataset from [Kaggle](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset).
2.  Preprocess the images (resize, normalize, equalize histograms). The `CBISDDSMDataset` class in `utils.py` handles this.
3.  Organize your data into a CSV file with columns: `['image_path', 'label']`.

## Installation
1.  Clone this repo: `git clone https://github.com/your_username/breast-cancer-rhcc.git`
2.  Install dependencies: `pip install -r requirements.txt`

## Usage
1.  Update the paths in `train.py` to point to your data CSVs and image directory.
2.  Run the training script: `python train.py`
    - This will train the "Conventional CNN 3D" baseline model.
    - To train the "Conventional DL" (ResNet-18 3D) baseline, uncomment the relevant line in `train.py`.

## Results
The script will output metrics including Accuracy, Precision, Recall, F1-Score, and AUC on the validation set, providing a benchmark for comparison against the proposed RHCC method.

## Disclaimer
This code implements the comparative baselines and the core data pipeline described in the paper. The full RHCC pipeline, involving complex 3D reconstruction and active contour segmentation, is a significant research contribution and is not fully implemented here due to its complexity. The results in the paper are based on that complete implementation.

