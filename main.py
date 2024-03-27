
#%% Import statements

from datetime import datetime
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter


#%% LSTM Model - Example Torch model

hidden_size = 64
n_layers = 3
n_features = 18

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x


#%% Training and Validation

# Hyperparams
N_EPOCHS = 10000
BATCH_SIZE = 64
EVAL_PERIOD = 20
CHECKPOINT_PERIOD = 20
LR = 0.01 # 0.001

CHECKPOINT_DIR = "checkpoints"
MODEL_NAME = "lstm-model"

DEVICE = "cuda"
PIN_MEMORY = True
NUM_WORKERS = 4

# Set train and test sets -TODO MODIFY!
X_train, y_train = [], []
X_test, y_test = [], []

# Configure model
model = LSTMModel() # TODO MODIFY!
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9) #optim.Adam(model.parameters())
loss_fn = nn.MSELoss()

train_loader = data.DataLoader(data.TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
test_loader = data.DataLoader(data.TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

# Set checkpoint directory
checkpoint_dir = Path(CHECKPOINT_DIR, MODEL_NAME, datetime.now().strftime('%Y%m%d-%H%M%S'))
assert not os.path.isdir(checkpoint_dir), "Model checkpoint directory already exists"

os.makedirs(checkpoint_dir, exist_ok=True)

# Set Tensorboard Writer
log_dir = Path(checkpoint_dir, "logs/scalars")
writer = SummaryWriter(log_dir=log_dir)

# Display the model overview
print(model)

for layer in model.children():
    print("Layer : {}".format(layer))
    print("Parameters : ")
    for param in layer.parameters():
        print(param.shape)
    print()

model = model.to(DEVICE)

for epoch in range(N_EPOCHS):
    model.train()
    print("training...")

    running_train_loss = 0
    with tqdm(train_loader, unit="batch") as tepoch: # show progress bar for batches
        for idx, batch in enumerate(tepoch):
            tepoch.set_description(f"Epoch {epoch}")

            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tepoch.set_postfix(loss=loss.item())

            running_train_loss += loss

        avg_train_rmse = torch.sqrt(running_train_loss / (idx + 1))
        writer.add_scalar("Loss/train_rmse", avg_train_rmse, epoch)
        print(f"Epoch {epoch}: train RMSE {avg_train_rmse:.7f}")

        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if epoch % CHECKPOINT_PERIOD == CHECKPOINT_PERIOD - 1:
            ckpt_path = Path(checkpoint_dir, MODEL_NAME + "_epoch" + str(epoch) + ".pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, ckpt_path)

        # Validation
        if epoch % EVAL_PERIOD == EVAL_PERIOD - 1:

            model.eval()
            running_test_loss = 0
            print('validating...')
            
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():

                for idx, batch in enumerate(test_loader):
                    X_test, y_test = batch
                    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
                    y_pred = model(X_test)
                    test_loss = loss_fn(y_pred, y_test)
                    running_test_loss += test_loss
            
                avg_test_rmse = torch.sqrt(running_test_loss / (idx + 1))
            
            print(f"Epoch {epoch}: test RMSE {avg_test_rmse:.7f}")
            writer.add_scalar("Loss/validation_rmse", avg_test_rmse, epoch)

writer.flush()
writer.close()


# %% Load checkpoint

model = LSTMModel()

checkpoint = torch.load(ckpt_path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# eval mode to set dropout and batch normalization layers to evaluation mode before inference. 
# Failing to do this will yield inconsistent inference results.
model.eval()
