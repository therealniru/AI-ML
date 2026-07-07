import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import optuna


"""
FashionMNIST Image Classification with PyTorch and Optuna Hyperparameter Tuning

This project implements a fully connected Artificial Neural Network (ANN) using 
PyTorch to classify the FashionMNIST dataset. It utilizes Optuna for automated 
hyperparameter optimization to discover the optimal network architecture and 
training configuration.

Key Features:
-------------
- Dataset: FashionMNIST (28x28 grayscale images of clothing items across 10 classes)
- Framework: PyTorch (Neural network architecture, training loops, and data loaders)
- Optimization: Optuna (Automated search for optimal hyperparameters)

Hyperparameters Tuned by Optuna:
--------------------------------
- Number of hidden layers and the number of hidden units per layer
- Learning rate (lr) for the gradient descent optimizer
- Weight decay (L2 regularization coefficient)
- Choice of optimizer (e.g., Adam, SGD, RMSprop)
- Dropout probabilities for preventing overfitting

Usage:
------
Ensure your virtual environment is active and all dependencies from 
'requirements.txt' are installed before executing this script.
"""

# Loading the dataset
df=pd.read_csv("fashion-mnist_train.csv")
# print(df.head())


#Spliting data to features and labels
X=df.iloc[:,1:].values
y=df.iloc[:,0].values
# print(X,y)

# Make training and testing data 80% training data and 20% test data
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42) # random_state is set for reproducibility
# print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

# Normalise X_train and X_test
X_train=X_train/255.0
X_test=X_test/255.0

# CustomDataset class to use Dataset and DataLoader (helps with making batches while training)

class CustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features=torch.tensor(features,dtype=torch.float32)
        self.labels=torch.tensor(labels,dtype=torch.long)
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self,index):
        return self.features[index], self.labels[index]

train_dataset= CustomDataset(X_train,y_train)
test_dataset= CustomDataset(X_test,y_test)
# print(train_dataset[10],len(train_dataset), test_dataset[10], len(test_dataset))

# Making NN class
class FMNIST_NN(nn.Module):
    def __init__(self,input_dim,output_dim,num_hidden_layers,neurons_per_layer,dropout_rate):
        super().__init__()

        layers=[]   # Linear Layer ➔ Batch Norm ➔ ReLU Activation ➔ Dropout

        for layer in range(num_hidden_layers):
            layers.append(nn.Linear(input_dim,neurons_per_layer))
            layers.append(nn.BatchNorm1d(neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim=neurons_per_layer
        layers.append(nn.Linear(neurons_per_layer,output_dim))

        self.model=nn.Sequential(*layers) # *args to unpack list to layer as Sequential expects layer(s) explicitly

    def forward(self,features):
        return self.model(features)


# Making the objective function for Optuna to optimize all hyperparams
def objective(trial):
    # Making all hyperparams which will be used my model

    num_hidden_layers=trial.suggest_int("num_hidden_layers",1,5)
    neurons_per_layer=trial.suggest_int("neurons_per_layer",8,128,step=8) # Start from value 8 till 128
    epochs=trial.suggest_int("epochs",10,100,step=10) # Scale with step size 10
    lr=trial.suggest_float("lr",1e-5,1e-1,log=True) # Scale in log terms not linearly
    dropout_rate=trial.suggest_float("dropout_rate",0.1,0.5,step=0.1)
    batch_size=trial.suggest_categorical("batch_size",[16,32,64,128])
    op=trial.suggest_categorical("op",["Adam","SGD","RMSprop"])
    weight_decay=trial.suggest_float("weight_decay",1e-5,1e-3,log=True)

    # Defining the training loop

    input_dim=X_train.shape[1]
    output_dim=10

    loss_fn=nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Device: {device}")

    model=FMNIST_NN(input_dim,output_dim,num_hidden_layers,neurons_per_layer,dropout_rate)
    model=model.to(device) # Moving to cuda or gpu or mps (apple silicon)

    if op == "Adam":
        op=optim.Adam(model.parameters(),lr=lr, weight_decay=weight_decay)
    elif op == "SGD":
        op=optim.SGD(model.parameters(),lr=lr, weight_decay=weight_decay)
    else:
        op=optim.RMSprop(model.parameters(),lr=lr, weight_decay=weight_decay)

    train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,pin_memory=True)
    test_dataloader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False,pin_memory=True)

    print(f"Starting Training for Trial: {trial}")

    for epoch in range(epochs):
        for batch_features, batch_labels in train_dataloader:

            batch_features,batch_labels=batch_features.to(device), batch_labels.to(device)

            y_preds=model(batch_features) # Forward Pass

            loss=loss_fn(y_preds,batch_labels) # Find Loss

            op.zero_grad()  # Avoids grad accumulation

            loss.backward() # Finds dL/dw and dL/db

            op.step()

        if epoch % 10 == 0:
            print(f"Epoch: {epoch} Loss: {loss:.3f}")

    print(f"Training Done for Trial: {trial}")
    # print(batch_features.shape,batch_labels.shape)
    model.eval() # Set model in eval mode

    correct=0
    total=0 # To find accuracy of model
    with torch.no_grad(): # Turn off grads
        for batch_features,batch_labels in test_dataloader:
            batch_features,batch_labels=batch_features.to(device), batch_labels.to(device)
            y_test_preds= model(batch_features)
            _,preds=torch.max(y_test_preds,1)
            total+=batch_labels.shape[0]
            correct+= (preds==batch_labels).sum().item()

        accuracy=correct/total

    return accuracy


study=optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
n_trials=10 # Set based on device specifications and gpu availability (higher better)
study.optimize(objective,n_trials=n_trials) # Making optuna find best values for hyperparams

print(f"Best Value (Accuracy): {(study.best_value)*100:.3f}% Best Params: {study.best_params}")


























