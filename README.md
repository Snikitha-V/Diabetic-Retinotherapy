# Diabetic Retinopathy Detection

This project uses machine learning to detect diabetic retinopathy from retinal images, aiming to assist in early diagnosis and prevent vision loss.

The project combines deep learning for feature extraction with traditional machine learning models to classify retinal images into severity levels.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Models and Training](#models-and-training)
- [Results](#results)
- [Visualization](#visualization)
- [Saved Models](#saved-models)
- [Contributing](#contributing)
- [License](#license)

## Overview
Diabetic retinopathy (DR) is a leading cause of blindness. This project uses the mBRSET dataset to train machine learning models for detection. Features are extracted using ResNet50, reduced with PCA, and classified using various algorithms.

Pipeline:
1. Feature Extraction with ResNet50.
2. Dimensionality Reduction with PCA.
3. Model Training and Evaluation.
4. Visualization of Results.
5. Model Saving.

## Features
- Multi-model classification.
- ResNet50 feature extraction.
- PCA for dimensionality reduction.
- Model evaluation and comparison.
- Data visualization.
- Saved models for reuse.

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/Snikitha-V/Diabetic-Retinotherapy.git
   cd Diabetic-Retinotherapy
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place in `mBRSET Data/`.

## Dataset
Using the mBRSET Dataset from Kaggle, with retinal images in 5 classes.

- Source: [Kaggle mBRSET Dataset](https://www.kaggle.com/datasets/jaskiratsinghchopra/mbrset-data/discussion?sort=hotness)
- Structure: JPEG images in folders 0-4.
- Preprocessing: Resize to 224x224, normalize, 80/20 train/val split.

## Usage
Run the notebook `notebooks/train_mbrset_colab.ipynb` to train models and generate results.

## Models and Training
Models include Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest, XGBoost, AdaBoost, MLP, and CategoricalNB.

Training: Extract features, apply PCA, train models, evaluate on validation set.

## Results
Validation accuracies from the latest run:

| Model                      | Val Accuracy |
|----------------------------|--------------|
| Logistic Regression        | 0.7703      |
| SVM                        | 0.7692      |
| AdaBoost                   | 0.7692      |
| Random Forest              | 0.7692      |
| Naive Bayes (Gaussian)     | 0.7641      |
| MLP                        | 0.7621      |
| Bayesian Network (CategoricalNB) | 0.7621 |
| XGBoost                    | 0.7621      |
| KNN                        | 0.7569      |
| Decision Tree              | 0.6041      |

Full results in `notebooks/results_summary.csv`.

## Visualization
The project includes a horizontal bar chart comparing model accuracies, generated in the notebook using Matplotlib. The chart ranks models from highest to lowest accuracy, with bars colored in sky blue. For example, Logistic Regression appears at the top with ~77% accuracy, while Decision Tree is at the bottom with ~60%. Run the notebook to view the interactive chart.

## Saved Models
Models saved in `notebooks/saved_models/` using Joblib for future use.

## Contributing

Contributions welcome. Fork, make changes, submit PR.
