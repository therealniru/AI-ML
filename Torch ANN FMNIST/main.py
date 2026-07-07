text = '''
+-----------------------------------------------------------------------+
|                                                                       |
|                                WELCOME             
|                                                                       |
+-----------------------------------------------------------------------+

=========================================================================
FASHION-MNIST CLASSIFICATION WITH ANN & OPTUNA HYPERPARAMETER TUNING
=========================================================================

### 1. Project Overview
This project implements a fully connected Artificial Neural Network (ANN) using 
PyTorch to classify the Fashion-MNIST dataset. Instead of manually guessing the 
best network structure and training settings, we use Optuna—an automated, 
state-of-the-art hyperparameter optimization framework—to systematically find the 
most efficient model configuration.

### 2. The Core Intuition (How it Works)
Think of training a neural network like tuning a highly complex musical instrument. 
If you change one knob (like the learning rate), it affects how all the other knobs 
behave. 

Instead of randomly guessing or using a slow, brute-force grid search, Optuna acts 
like an intelligent assistant. It runs multiple "Trials," looks at how previous 
trials performed, and uses probabilistic algorithms to guess which knob combinations 
will yield a lower loss and higher accuracy.

### 3. Hyperparameters being Optimized
Optuna dynamically samples the following parameters during its search:
*   num_hidden_layers : The depth of the network (e.g., 1 to 3 layers).
*   neurons_per_layer : The width/capacity of each layer (e.g., 16 to 128 neurons).
*   epochs            : How many times the model sees the entire dataset.
*   lr (Learning Rate): How big of a step the optimizer takes during gradient descent.
*   dropout_rate      : The percentage of randomly deactivated neurons to prevent overfitting.

### 4. Current Execution Insights
From the terminal outputs, Optuna is actively evaluating trials on the 'mps' device 
(Apple Silicon GPU acceleration). 
* Example Trial Run: A network configuration with 1 hidden layer, 24 neurons, 
  70 epochs, and a learning rate of ~0.009 was tested, tracking its loss reduction 
  over time to evaluate performance.

=========================================================================
'''

print(text)
