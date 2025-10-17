# 🩺👁️ Diabetic Retinopathy Detection: Illuminating the Shadows of Vision 🕵️‍♂️

Welcome to the **Diabetic Retinopathy Detection** project! This is not just a machine learning endeavor; it's a quest to harness the power of artificial intelligence to safeguard one of humanity's most precious gifts—sight. By analyzing retinal images, we aim to detect diabetic retinopathy early, potentially preventing blindness and transforming lives. 🌟

Imagine a world where a simple image can unlock the secrets of eye health, where algorithms dance with data to reveal hidden threats. That's what this project is all about: blending cutting-edge deep learning with classical machine learning to create a robust, multi-model detection system. 🚀

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset](#dataset)
- [Usage](#usage)
- [Models and Training](#models-and-training)
- [Results](#results)
- [Saved Models](#saved-models)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview
Diabetic retinopathy (DR) is a leading cause of blindness worldwide, often sneaking up silently until it's too late. This project leverages the mBRSET dataset—a treasure trove of retinal images—to train and evaluate various machine learning models. We extract powerful features using ResNet50, a pre-trained convolutional neural network, then apply dimensionality reduction and deploy an ensemble of classifiers to achieve accurate detection.

Our pipeline is a symphony of steps:
1. **Feature Extraction**: ResNet50 transforms images into rich feature vectors.
2. **Dimensionality Reduction**: PCA condenses these features for efficiency.
3. **Model Training**: A lineup of ML heroes battles it out on the data.
4. **Evaluation**: Accuracy scores reveal the champions.
5. **Visualization**: Charts and graphs bring the results to life.
6. **Model Persistence**: Save the best performers for future quests.

## ✨ Features
- **Multi-Model Ensemble**: From Logistic Regression to XGBoost, we've got a full roster of classifiers.
- **Deep Learning Integration**: ResNet50 for feature extraction—because why reinvent the wheel?
- **Dimensionality Magic**: PCA reduces complexity without losing essence.
- **Bayesian Networks**: A nod to probabilistic modeling with CategoricalNB.
- **Visualization Galore**: Bar charts to compare model performances.
- **Model Saving**: Joblib-powered persistence for easy deployment.
- **Caching**: Smart feature caching to save time on reruns.
- **Git-Friendly**: .gitignore ensures only the essentials are tracked.

## 🛠️ Installation
Ready to dive in? Follow these steps to set up your environment:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Snikitha-V/Diabetic-Retinotherapy.git
   cd Diabetic-Retinotherapy
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.7+ installed. Then:
   ```bash
   pip install -r requirements.txt
   ```
   This will install all necessary libraries, including TensorFlow, scikit-learn, XGBoost, and more.

3. **Download the Dataset**:
   - The mBRSET dataset should be placed in the `mBRSET Data/` directory.
   - Organize it into subfolders (0/, 1/, 2/, 3/, 4/) as expected by the ImageDataGenerator.
   - Note: The dataset is large and not included in the repo—download it from the official source.

4. **Run the Notebook**:
   - Open `notebooks/train_mbrset_colab.ipynb` in Jupyter or VS Code.
   - Execute cells step by step to train models and generate results.

## 📊 Dataset
We're using the **mBRSET Dataset**, a meticulously curated collection of retinal images for diabetic retinopathy research. It includes images categorized into 5 classes (0-4), representing varying severity levels.

- **Source**: [Kaggle mBRSET Dataset](https://www.kaggle.com/datasets/jaskiratsinghchopra/mbrset-data/discussion?sort=hotness)
- **Structure**: Images are in JPEG format, organized by class in numbered folders.
- **Preprocessing**: Images are resized to 224x224, normalized, and split into training/validation sets (80/20).
- **Size**: Thousands of images—perfect for robust training!

## 🚀 Usage
1. **Prepare Data**: Ensure the dataset is in place.
2. **Run Feature Extraction**: The notebook extracts features using ResNet50 and caches them as .npy files.
3. **Train Models**: Watch as various ML models are trained on PCA-reduced features.
4. **Evaluate**: Check accuracy scores and visualize results.
5. **Save Models**: Models are saved in `notebooks/saved_models/` for later use.

To run the entire pipeline:
- Open the notebook and run all cells.
- Results will be saved to `notebooks/results_summary.csv`.
- Visualizations will display inline.

## 🤖 Models and Training
Our arsenal includes:
- **Logistic Regression**: The reliable workhorse.
- **K-Nearest Neighbors (KNN)**: Proximity-based prediction.
- **Support Vector Machine (SVM)**: Kernel-powered classification.
- **Naive Bayes (Gaussian)**: Probabilistic simplicity.
- **Decision Tree**: Intuitive rule-based learning.
- **Random Forest**: Ensemble of trees for robustness.
- **XGBoost**: Gradient boosting champion.
- **AdaBoost**: Adaptive boosting wizard.
- **MLP (Multi-Layer Perceptron)**: Neural network finesse.
- **Bayesian Network (CategoricalNB)**: For discretized features.

Training involves:
- Feature extraction from ResNet50.
- PCA to 20 components.
- Discretization for Bayesian models.
- Fitting and prediction on validation set.

## 📈 Results
After training, we compare validation accuracies. Here's a sneak peek (based on sample runs):

| Model                      | Val Accuracy |
|----------------------------|--------------|
| Random Forest             | 0.85        |
| XGBoost                   | 0.83        |
| SVM                       | 0.82        |
| ...                       | ...         |

Visualizations include horizontal bar charts ranking models by accuracy. Full results are in `notebooks/results_summary.csv`.

## 💾 Saved Models
Trained models are saved using Joblib in `notebooks/saved_models/`:
- `Logistic_Regression.joblib`
- `Random_Forest.joblib`
- And more!

Load them for inference:
```python
import joblib
model = joblib.load('notebooks/saved_models/Random_Forest.joblib')
predictions = model.predict(X_test)
```

## 🤝 Contributing
Join the vision-saving mission! Contributions are welcome:
- Fork the repo.
- Create a feature branch.
- Make your changes.
- Submit a pull request.

Ideas for improvement:
- Add more models (e.g., LightGBM).
- Implement cross-validation.
- Deploy as a web app with Streamlit.

## 📄 License
This project is licensed under the MIT License—free to use, modify, and distribute. See `LICENSE` for details.

---

*Crafted with ❤️ by the Diabetic Retinopathy Detection Team. Let's turn data into destiny and code into cures!* 🌈