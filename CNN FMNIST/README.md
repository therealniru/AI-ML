# FashionMNIST Classification with PyTorch CNN

This project implements a deep Convolutional Neural Network (CNN) using PyTorch to classify the FashionMNIST dataset. The model utilizes spatial feature extraction via localized convolutions followed by dense fully connected layers to map raw image inputs to discrete clothing category predictions.

## 🚀 Key Features

- **Sequential Spatial Feature Extraction**: Implements an optimized pipeline featuring `Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d` architectural blocks to extract multi-scale spatial features.
- **Robust Regularization Structure**: Integrates targeted spatial downsampling alongside dual high-probability Dropout layers (0.4) inside the classifier head to penalize model complexity and combat overfitting.
- **Normalized Gradient Stability**: Utilizes batch normalization layers directly following convolutions to stabilize intermediate activations, accelerate overall training speed, and mitigate gradient vanishing.
- **Cross-Platform Device Selector**: Optimized execution layout that automatically binds to `cuda`, Apple Silicon `mps`, or falls back gracefully to `cpu`.

## Project Setup & Installation

Follow these steps to configure your project workspace across macOS, Linux, or Windows environments.

### 1. Clone the Repository

```bash
git clone https://github.com/therealniru/AI-ML.git
cd "Torch CNN FMNIST"
```

### 2. Configure and Activate the Virtual Environment (venv)

**🍏 On macOS / Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**🪟 On Windows:**

```bash
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all required platform packages listed in the requirements manifest:

```bash
pip install -r requirements.txt
```

### 4. PyCharm IDE Interpreter Synchronization

To link the PyCharm editor interface with your active environment configuration across platforms:

1. Select the Python interpreter status widget located in the bottom-right corner of the window.
2. Select **Add New Interpreter → Add Local Interpreter...**
3. Navigate to **Virtualenv Environment** in the sidebar options and select the **Select existing environment** radio option.
4. Set the target file browser path pointer to:
   - `venv/bin/python` (macOS / Linux platforms)
   - `venv/Scripts/python.exe` (Windows platforms)
5. Click **OK** to begin package mapping and clear missing requirement banners.

## 📦 Dataset Acquisition (Kaggle)

The model pipeline is configured to parse the tabular variant of the dataset.

1. Download the `fashion-mnist_train.csv` file directly from the Kaggle Fashion-MNIST Hub.
2. Save the extracted `.csv` dataset file into the root of your project directory where your script resides.

## 💻 Running the Project

To execute the data preprocessing, model compilation, training loop execution, and test evaluation sequences directly from your active terminal session, run:

```bash
python CNN.py
```