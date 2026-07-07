# Project Setup & Installation



Follow these steps to configure your project workspace across macOS, Linux, or Windows environments.



## 1. Clone the Repository

```bash

git clone https://github.com/therealniru/AI-ML.git

cd "Torch ANN FMNIST"

```



## 2. Configure and Activate the Virtual Environment (`venv`)



### 🍏 On macOS / Linux:

```bash

python3 -m venv venv

source venv/bin/activate

```



### 🪟 On Windows:

```bash

python -m venv venv

.\venv\Scripts\activate

```



## 3. Install Dependencies

Install all required platform packages listed in the requirements manifest:

```bash

pip install -r requirements.txt

```



## 4. PyCharm IDE Interpreter Synchronization

To link the PyCharm editor interface with your active environment configuration across platforms:

1. Select the Python interpreter status widget located in the bottom-right corner of the window.

2. Select **Add New Interpreter** → **Add Local Interpreter...**

3. Navigate to **Virtualenv Environment** in the sidebar options and select the **Select existing** environment radio option.

4. Set the target file browser path pointer to:

   - `venv/bin/python` (macOS / Linux platforms)

   - `venv/Scripts/python.exe` (Windows platforms)

5. Click **OK** to begin package mapping and clear missing requirement banners.



---



# FashionMNIST Classification with PyTorch & Optuna ANN



This project implements a fully connected Artificial Neural Network (ANN) using **PyTorch** to classify the **FashionMNIST** dataset. The project integrates **Optuna** for automated hyperparameter tuning to dynamically discover the optimal network architecture, regularization layers, and optimizer configurations.



## 🚀 Key Features

- **Dynamic Network Architecture:** Supports a variable number of hidden layers and hidden units per layer, dynamically constructed using `nn.Sequential`.

- **Advanced Regularization:** Integrates configurable `Linear -> Batch Norm -> ReLU -> Dropout` blocks within hidden layers to enhance gradient flow and minimize overfitting.

- **Automated Hyperparameter Optimization:** Uses Optuna to tune:

  - Number of hidden layers and layer widths.

  - Learning rate (`lr`) on a logarithmic scale.

  - Optimizer selection (`Adam`, `SGD`, `RMSprop`).

  - Dropout rates and optimization epochs.

- **Cross-Platform Device Selector:** Optimized execution layout that automatically binds to `cuda`, Apple Silicon `mps`, or falls back gracefully to `cpu`.



## 💻 Running the Project

To execute the hyperparameter optimization script and launch the tuning trials directly from your active terminal session, run:

```bash

python ANN.py

```