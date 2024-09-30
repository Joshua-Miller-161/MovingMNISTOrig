import sys
sys.dont_write_bytecode = True
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchinfo import summary
import argparse

sys.path.append(os.getcwd())
from misc import *
from models import ConvLSTM, Seq2Seq
#====================================================================
device = GetDevice()
torch.set_default_device(device)
#====================================================================
''' Data '''

path = 'mnist_test_seq.npy'

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--path', type=str, nargs=1,
                    help='An optional integer positional argument', required=False)
args = parser.parse_args()
arg_path = args.path

try:
    if (len(arg_path) > 0):
        path = arg_path[0]
except TypeError:
    pass

MovingMNIST = np.load(path).transpose(1, 0, 2, 3)
print(" >> MovingMNIST:", MovingMNIST.shape)

# Shuffle Data
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]         
val_data = MovingMNIST[8000:9000]       
test_data = MovingMNIST[9000:10000]     

def collate(batch):

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)     
    batch = batch / 255.0                        
    batch = batch.to(device)                     

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)                     
    return batch[:,:,rand-10:rand], batch[:,:,rand]     

# Training Data Loader
train_loader = DataLoader(train_data, shuffle=True, 
                          batch_size=16, collate_fn=collate,
                          generator=torch.Generator(device=device))

# Validation Data Loader
val_loader = DataLoader(val_data, shuffle=True, 
                        batch_size=16, collate_fn=collate, 
                        generator=torch.Generator(device=device))

x_init, y_init = next(iter(train_loader))
print(" >> x_init", x_init.shape, x_init.device, ", y_init", y_init.shape, y_init.device)
#====================================================================
''' Model '''

# model = ConvLSTM(in_channels=1, 
#                  out_channels=3,
#                  kernel_size=(3, 3),
#                  padding='same', 
#                  activation="tanh",
#                  frame_size=(x_init.shape[3], x_init.shape[4])).to(device)

model = Seq2Seq(num_channels=1,
                num_kernels=64, 
                kernel_size=(3, 3),
                padding=(1, 1),
                activation="tanh", 
                frame_size=(x_init.shape[3], x_init.shape[4]),
                num_layers=3).to(device)

optim     = Adam(model.parameters(), lr=1e-4)
criterion = nn.BCELoss(reduction='sum')

x_test = torch.rand_like(x_init)
y_test = model(x_test)
print(" >> y_test", y_test.shape, y_test.device)

print(" >> Devices: model", next(model.parameters()).device, ", x", x_init.device, ", y", y_init.device)

summary(model, input_size=x_init.shape, col_names=("input_size", "output_size", "num_params"), verbose=1, depth=7, device=device)

#plot_model(model, x_init.shape, model_name='MovingMNIST', device=device)

#====================================================================
num_epochs = 20

for epoch in range(1, num_epochs+1):
    
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        output = model(input)                                     
        loss = criterion(output.flatten(), target.flatten())       
        loss.backward()                                            
        optim.step()                                               
        optim.zero_grad()                                           
        train_loss += loss.item()                                 
    train_loss /= len(train_loader.dataset)                       

    val_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in val_loader:                          
            output = model(input)                                   
            loss = criterion(output.flatten(), target.flatten())   
            val_loss += loss.item()                                
    val_loss /= len(val_loader.dataset)                            

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))


#====================================================================


#====================================================================


#====================================================================