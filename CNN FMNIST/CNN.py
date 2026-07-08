import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Loading data from dataset
df=pd.read_csv("fashion-mnist_train.csv")
# print(df.head())

# Getting features and labels
X=df.iloc[:,1:].values
y=df.iloc[:,0].values
# print(X,y,X.shape,y.shape)

# Making training and testing data 80-20 split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) # For reproducibility
X_train=X_train/255.0
X_test=X_test/255.0 # Scaling for ease of training
print("Shapes of training and testing data:")
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

# Setting appropriate GPU for running code

if torch.cuda.is_available():
    device="cuda"
elif torch.backends.mps.is_available():
    device="mps"
else:
    device="cpu"

print(f"Device: {device}")

class MyCustomDataset(Dataset):
    def __init__(self,features,labels):
        self.features=torch.tensor(features,dtype=torch.float32).reshape(-1,1,28,28) # As we are making CNN and image is 28 x 28 and grayscale -1 is for batches
        self.labels=torch.tensor(labels,dtype=torch.long)
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self,index):
        return self.features[index],self.labels[index]



# Making dataset and dataloader objects
train_dataset=MyCustomDataset(X_train,y_train)
test_dataset=MyCustomDataset(X_test,y_test)

train_dataloader=DataLoader(train_dataset,batch_size=32,shuffle=True,pin_memory=True)
test_dataloader=DataLoader(test_dataset,batch_size=32,shuffle=False,pin_memory=True)

# index=8
# print(train_dataloader[index],len(train_dataloader))
# print(test_dataloader[index],len(test_dataloader))
# print(train_dataloader.shape,test_dataloader.shape))

# Making the CNN
# Pipeline: Input (grayscale) -> Conv -> BatchNorm -> ReLU -> MaxPool -> Conv -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> Output
class FMNIST_CNN(nn.Module):
    def __init__(self,num_channels):

        super().__init__()
        self.convolutional=nn.Sequential(
            nn.Conv2d(num_channels,32,kernel_size=3,padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),

            nn.Conv2d(32,64,kernel_size=3,padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.fc=nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64,10)
        )


    def forward(self,features):
        return self.fc(self.convolutional(features))



# Setting hyperparameters and Loss and Optimizer, please adjust according to GPU availability

lr=0.1
epochs=100
weight_decay=1e-4
num_channels=1 # As grayscale image

model=FMNIST_CNN(num_channels)
model=model.to(device)

loss_fn=nn.CrossEntropyLoss()
op=optim.SGD(model.parameters(),lr=lr,weight_decay=weight_decay) # For Regularization

# Training loop
print(f"Starting training for {epochs} epochs")
for epoch in range(epochs):
    for batch_features, batch_labels in train_dataloader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
        y_pred=model(batch_features) # Forward pass
        loss=loss_fn(y_pred,batch_labels) # Find loss
        op.zero_grad() # Zero out gradients to avoid accumulation of gradients
        loss.backward() # Finds dL/dw and dL/db
        op.step() # Does the gradient descent w-=lr*dL/dw and b-=lr*dL/db

    if epoch % 10 == 0:
        print(f"Epoch: {epoch} Loss: {loss}")


print("Training Done")

print("Evaluating Model")
# Evaluate model
model.eval()

correct=0
total=0
with torch.no_grad():
    for batch_features,batch_labels in test_dataloader:
        batch_features, batch_labels= batch_features.to(device),batch_labels.to(device)
        y_test_preds=model(batch_features) # Forward pass on test (unseen) data
        preds=torch.argmax(y_test_preds,dim=1) # Finding class number of item
        total+=batch_features.shape[0]
        correct+= (preds==batch_labels).sum().item() # Finding right labels based on batch_labels

    print(f"Accuracy: {(correct/total) * 100 :.3f}%") # Calculating Accuracy

print("Evaluation Done")



