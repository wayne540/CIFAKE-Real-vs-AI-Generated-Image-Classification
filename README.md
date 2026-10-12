# CIFAKE: Real vs AI-Generated Image Classification

A deep learning project that uses a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images as either **REAL** or **AI-GENERATED**.

The project explores how convolutional neural networks can learn visual patterns that distinguish real images from synthetic images generated using artificial intelligence.

## Table of Contents

* [Project Overview](#project-overview)
* [Objectives](#objectives)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Data Preprocessing](#data-preprocessing)
* [Training Strategy](#training-strategy)
* [Model Evaluation](#model-evaluation)
* [Technologies Used](#technologies-used)
* [Installation](#installation)
* [Running the Project](#running-the-project)
* [Making Predictions](#making-predictions)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [References](#references)

## Project Overview

AI-generated images are becoming increasingly realistic, making it more difficult to distinguish synthetic images from real photographs.

This project investigates the use of deep learning for binary image classification. A custom CNN is trained on the CIFAKE dataset to learn features that help distinguish real images from AI-generated synthetic images.

The implementation uses TensorFlow and Keras to build the neural network, prepare the dataset, train the model, and evaluate its performance on previously unseen test images.

The project also explores data augmentation, early stopping, and multiple classification metrics to develop a more complete understanding of the image classification workflow.

## Objectives

The main objectives of this project are to:

* Develop a CNN for binary image classification.
* Understand how convolutional layers extract visual features from images.
* Apply image preprocessing and data augmentation.
* Use a validation dataset to monitor generalization during training.
* Implement early stopping to reduce unnecessary training.
* Evaluate model performance using multiple classification metrics.
* Generate predictions for individual images.
* Investigate the limitations of CNN-based AI-generated image detection.

## Dataset

**Dataset:** CIFAKE: Real and AI-Generated Synthetic Images

**Dataset source:** [Kaggle CIFAKE Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

The dataset contains 120,000 RGB images belonging to two classes: REAL and FAKE.

| Property                     | Description           |
| ---------------------------- | --------------------- |
| Total images                 | 120,000               |
| Training images              | 100,000               |
| Testing images               | 20,000                |
| Real training images         | 50,000                |
| AI-generated training images | 50,000                |
| Real testing images          | 10,000                |
| AI-generated testing images  | 10,000                |
| Image dimensions             | 32 × 32 pixels        |
| Image channels               | 3 (RGB)               |
| Classification type          | Binary classification |

### Dataset Classes

**REAL:** Images sourced from the CIFAR-10 dataset.

**FAKE:** Synthetic images generated using Stable Diffusion 1.4.

The dataset is balanced, with equal numbers of real and synthetic images in the training and testing sets.

### Dataset Organization

The project expects the extracted dataset to follow this directory structure:

```text
cifake_data/
├── train/
│   ├── REAL/
│   └── FAKE/
└── test/
    ├── REAL/
    └── FAKE/
```

The dataset is not included in this repository. Download it separately from Kaggle and extract it before running the notebook.

## Model Architecture

The project implements a custom Convolutional Neural Network using TensorFlow and Keras.

The architecture consists of three convolutional blocks, followed by global average pooling, dropout, and a sigmoid output layer.

### Architecture Summary

| Layer                  | Configuration                          | Purpose                                      |
| ---------------------- | -------------------------------------- | -------------------------------------------- |
| Input                  | 32 × 32 × 3                            | Accepts RGB images                           |
| Data augmentation      | Random horizontal flip and translation | Introduces training variation                |
| Rescaling              | Pixel values divided by 255            | Normalizes image values                      |
| Conv2D                 | 32 filters, 2 × 2 kernel               | Extracts initial visual features             |
| MaxPooling2D           | Default pool size                      | Reduces spatial dimensions                   |
| Conv2D                 | 64 filters, 3 × 3 kernel               | Learns more complex features                 |
| MaxPooling2D           | Default pool size                      | Reduces spatial dimensions                   |
| Conv2D                 | 128 filters, 3 × 3 kernel              | Extracts higher-level features               |
| MaxPooling2D           | Default pool size                      | Further reduces spatial dimensions           |
| GlobalAveragePooling2D | Spatial averaging                      | Compresses feature maps                      |
| Dropout                | 0.3                                    | Helps reduce overfitting                     |
| Dense                  | 1 neuron, sigmoid activation           | Produces a binary classification probability |

### Output Interpretation

The model uses a sigmoid activation function to produce a probability between 0 and 1.

With the class ordering used in this project:

* **Probability below 0.50:** Predicted as REAL.
* **Probability greater than or equal to 0.50:** Predicted as AI-GENERATED.

The default decision threshold is 0.50, although it can be adjusted when appropriate.

The output represents the model's estimated probability for the AI-GENERATED class. It should not automatically be interpreted as a perfectly calibrated measure of confidence.

## Data Preprocessing

The dataset is loaded using TensorFlow's `image_dataset_from_directory` utility.

The preprocessing pipeline includes:

1. **Image resizing:** Images are resized to 32 × 32 pixels.
2. **Label inference:** Labels are assigned from the directory names.
3. **Binary encoding:** REAL is assigned label 0 and FAKE is assigned label 1.
4. **Training-validation split:** Twenty percent of the training directory is reserved for validation.
5. **Pixel normalization:** Image pixel values are rescaled from approximately 0–255 to 0–1.
6. **Data augmentation:** Random horizontal flips and small translations are applied during training.
7. **Dataset prefetching:** TensorFlow prefetching is used to help overlap data preparation with model execution.

### Dataset Splits

The training directory contains 100,000 images. With a validation split of 20%, the resulting datasets contain:

* **Training:** 80,000 images.
* **Validation:** 20,000 images.
* **Testing:** 20,000 images.

The test dataset remains separate from training and validation so that final performance can be evaluated on held-out data.

## Training Strategy

The model is compiled using the Adam optimizer and binary cross-entropy loss.

### Training Configuration

| Parameter               | Value                |
| ----------------------- | -------------------- |
| Optimizer               | Adam                 |
| Initial learning rate   | 0.001                |
| Loss function           | Binary Cross-Entropy |
| Maximum epochs          | 20                   |
| Batch size              | 128                  |
| Validation split        | 20%                  |
| Dropout rate            | 0.30                 |
| Early stopping patience | 3 epochs             |

### Early Stopping

Early stopping monitors the validation loss during training.

Training stops when validation loss fails to improve for three consecutive epochs. The callback restores the model weights from the epoch with the best validation loss.

This helps avoid unnecessary training and can reduce overfitting.

### Training Visualization

The notebook plots training and validation accuracy and loss over the completed epochs.

These plots help investigate:

* Whether the model is learning useful patterns.
* Whether training and validation performance diverge.
* Whether the model may be overfitting.
* Whether additional experiments with the architecture or hyperparameters are justified.

## Model Evaluation

The model is evaluated on the held-out test dataset.

The evaluation process includes several metrics to provide a broader view of classification performance.

| Metric           | Purpose                                                                                |
| ---------------- | -------------------------------------------------------------------------------------- |
| Accuracy         | Measures the proportion of correctly classified images                                 |
| Precision        | Measures how many images predicted as AI-generated are actually AI-generated           |
| Recall           | Measures how many AI-generated images are correctly identified                         |
| F1-score         | Combines precision and recall using their harmonic mean                                |
| ROC-AUC          | Measures how well the model separates the two classes across classification thresholds |
| Confusion matrix | Shows correct and incorrect predictions for both classes                               |

The notebook also generates a classification report and a confusion matrix to help identify the types of errors made by the model.

**Results:** Actual test metrics should be added here after training and evaluation are complete. No performance figures are claimed in this README before they have been measured.

## Technologies Used

* **Python:** Programming language.
* **TensorFlow:** Deep learning framework.
* **Keras:** Neural network construction and training API.
* **NumPy:** Numerical operations.
* **Matplotlib:** Visualization of training results and image predictions.
* **Seaborn:** Confusion matrix visualization.
* **Scikit-learn:** Classification metrics and evaluation.
* **Jupyter Notebook:** Interactive development and experimentation.
* **Google Colab:** Optional cloud environment for GPU-assisted training.

## Installation

### 1. Clone the Repository

Replace `<your-username>` and `<repository-name>` with your GitHub username and repository name.

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn jupyter
```

Depending on your hardware and TensorFlow version, local GPU support may require additional configuration.

## Running the Project

### Option 1: Google Colab

Google Colab is a convenient option when your computer does not have sufficient resources for model training.

1. Open [Google Colab](https://colab.research.google.com/).
2. Upload the project notebook.
3. Select **Runtime → Change runtime type**.
4. Select a GPU runtime if one is available.
5. Upload the dataset ZIP file or retrieve the dataset from your chosen storage location.
6. Update the dataset paths in the notebook.
7. Run the notebook cells in order.

To check whether TensorFlow can detect a GPU, run:

```python
import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))
```

GPU availability depends on the runtime and current resource limits.

### Option 2: Run Locally

After installing the dependencies:

1. Download the dataset from Kaggle.
2. Extract the ZIP file.
3. Ensure the directory structure matches the expected layout.
4. Update the dataset paths in the notebook.
5. Open the notebook in Jupyter or VS Code.
6. Execute the cells in order.

Training time depends on your hardware and the resources available to TensorFlow.

## Making Predictions

After training, the model can be saved in Keras format:

```python
model.save("cifake_detector.keras")
```

The saved model can then be loaded in a later session:

```python
import tensorflow as tf

model = tf.keras.models.load_model("cifake_detector.keras")
```

The notebook includes a `predict_image()` function for classifying individual images.

The function loads an image, converts it to RGB, resizes it to 32 × 32 pixels, prepares the input batch, and obtains a prediction from the trained model.

Example:

```python
result = predict_image("path/to/image.jpg")

print(result)
```

Replace the example path with the actual location of the image.

The model's internal rescaling layer handles pixel normalization, so the prediction function should not divide the input image by 255 a second time.

## Limitations

This project has several important limitations.

### 1. Low Image Resolution

The CIFAKE images are only 32 × 32 pixels. This limits the visual detail available to the model and differs substantially from the resolution of most modern photographs.

### 2. Limited Generator Diversity

The synthetic images in this dataset were generated using Stable Diffusion 1.4. Performance on images produced by other generative models may differ considerably.

### 3. Generalization

A model that performs well on CIFAKE may not perform equally well on images collected from unrelated sources or generated using newer systems.

### 4. Dataset-Specific Patterns

The model may learn patterns specific to the dataset rather than general indicators of synthetic image generation.

### 5. Probability Calibration

A sigmoid output is not necessarily a calibrated probability. Additional calibration and evaluation would be required before treating model scores as reliable real-world confidence estimates.

Consequently, this project should be treated as an educational image-classification experiment rather than a definitive AI-image verification system.

## Future Improvements

Potential extensions include:

* Comparing different CNN architectures.
* Experimenting with additional convolutional blocks and regularization techniques.
* Evaluating alternative learning rates and batch sizes.
* Investigating class-specific precision and recall.
* Plotting ROC and precision-recall curves.
* Applying explainability methods such as Grad-CAM.
* Testing the model on images from different sources and generative models.
* Investigating transfer learning and higher-resolution images.
* Developing an interactive application for image classification.
* Exploring probability calibration and more robust evaluation methods.

## References

### Dataset

Bird, J. J., & Lotfi, A. (2024). *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images.*

Dataset: [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)

Research paper: [IEEE Access](https://ieeexplore.ieee.org/abstract/document/10409290)

### Original CIFAR-10 Dataset

Krizhevsky, A., & Hinton, G. (2009). *Learning Multiple Layers of Features from Tiny Images.*

[University of Toronto technical report](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)

## Disclaimer

This project is intended for educational and research purposes. Predictions should not be treated as conclusive evidence that an image is real or AI-generated.

Model performance depends on the dataset, training procedure, image characteristics, and similarity between training data and images encountered in practice.
