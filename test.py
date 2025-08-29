import os
import pandas as pd
import numpy as np

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# input files
test_file = './test.csv'
train_file = './train.csv'

# hyperparameters
learning_rate = 5e-4
batch_size = 32
epochs = 15

# define dataset class
class PacketVolumeDataset(Dataset):

   def __init__(self, csv_file):
      # load csv to dataframe
      self.df = pd.read_csv(csv_file,header=None)

      # split features from label
      x = self.df.iloc[:, :-1].values
      y = self.df.iloc[:, -1].values

      # scale features
      min_vals = x.min(axis=0) # np array of mins for each column
      max_vals = x.max(axis=0) # np array of max for each column
      x_scaled = (x - min_vals) / (max_vals - min_vals + 1e-9)

      # convert to tensors
      self.x = torch.tensor(x_scaled, dtype=torch.float32) # float features
      self.y = torch.tensor(y, dtype=torch.long) # integer label

   def __len__(self):
      return len(self.df)

   def __getitem__(self, idx):
      return (self.x[idx],self.y[idx])

# init datasets
test_dataset = PacketVolumeDataset(csv_file=test_file)
train_dataset = PacketVolumeDataset(csv_file=train_file)

# init dataloaders
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# define neural net
class NeuralNetwork(nn.Module):
   def __init__(self):
      super().__init__()
      self.linear_relu_stack = nn.Sequential(
         nn.Linear(24, 64),
         nn.ReLU(),
         nn.Linear(64,64),
         nn.ReLU(),
         nn.Linear(64, 1)
      )

   def forward(self, x):
      logits = self.linear_relu_stack(x)
      return logits   # raw scores (logits), sigmoid later

model = NeuralNetwork()

# loss functino and optimizer
num_pos = (train_dataset.y == 1).sum()
num_neg = (train_dataset.y == 0).sum()
pos_weight = torch.tensor(num_neg / num_pos, dtype=torch.float32)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = AdamW(model.parameters(), lr=learning_rate)

# define training loop
def train_loop(dataloader, model, loss_fn, optimizer):
   size = len(dataloader.dataset)
   model.train()
   for batch, (x, y) in enumerate(dataloader):
      # comppute prediction and loss
      pred = model(x).squeeze(1) # remove label
      loss = loss_fn(pred, y.float())

      # back prop
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if batch % 100 == 0:
         loss_val, current = loss.item(), batch * batch_size + len(x)
         print(f"loss: {loss_val:>7f}  [{current:>5d}/{size:>5d}]")

# define testing loop
def test_loop(dataloader, model, loss_fn):
   model.eval()
   size = len(dataloader.dataset)
   num_batches = len(dataloader)
   test_loss, correct = 0, 0

   with torch.no_grad():
      for x, y in dataloader:
         pred = model(x).squeeze(1) # remove label
         test_loss += loss_fn(pred, y.float()).item()
         probs = torch.sigmoid(pred)
         predicted = (probs > 0.3).long()
         correct += (predicted == y).sum().item()

   test_loss /= num_batches
   correct /= size
   print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# train
for t in range(epochs):
   print(f"Epoch {t+1}\n-------------------------------")
   train_loop(train_dataloader, model, loss_fn, optimizer)
   test_loop(test_dataloader, model, loss_fn)
print("Done!")

# test
sample = np.array([3123275,3078842,2766476,2799381,2563540,2571852,2671010,2988791,2896058,2729306,2575240,2647077,2641776,2732868,2751066,2679022,2705588,2614844,2621927,3107265,2916175,2884927,2789264,2494150])

x_sample = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)

model.eval()

with torch.no_grad():
   logits = model(x_sample).squeeze(1)
   prob = torch.sigmoid(logits)
   predicted_class = (prob > 0.3).long()

print(f"Predicted class: {predicted_class.item()}")
