import torch.nn as nn
import numpy as np
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

seed = 42
torch.seed = seed

###  Separate  Gx==0 && Gy==0 from original csv ### 
### Since we need 5% Gx==0 && Gy==0 and all of Gx!=0 || Gy!=0 to balance the data. ###

# bgr_csv = pd.read_csv("./data/XYBGR.csv")
# grad_csv = pd.read_csv("./data/XYGxGy.csv")


# bgr_csv['Gx'] = grad_csv["Gx"]
# bgr_csv['Gy'] = grad_csv["Gy"]

# zeros_csv = bgr_csv[(bgr_csv['Gx']==0.0) & (bgr_csv['Gy']==0.0)]
# non_zeros_csv = bgr_csv[(bgr_csv['Gx']!=0.0) | (bgr_csv['Gy']!=0.0)]
# print(zeros_csv)
# print(non_zeros_csv)
# zeros_csv.to_csv("./data/zeros.csv",index=False)
# non_zeros_csv.to_csv("./data/nonzeros.csv",index=False)

###                   end                     ###

###                Get 5%                     ###
# zeros_csv = pd.read_csv("./data/zeros.csv")
# print(zeros_csv)
# print(zeros_csv['X'][0])
# nonzeros_csv = pd.read_csv("./data/nonzeros.csv")
# zeros_csv = zeros_csv.sample(frac=0.05)
# data = zeros_csv.append(nonzeros_csv) # all data we need
# data.to_csv("./data/cleaned_dataset.csv")
###                   end                     ###
### Split train and test ###
# data = pd.read_csv('./data/cleaned_dataset.csv')
# train_data = data.sample(frac=0.8,random_state=seed,axis=0)
# test_data = data[~data.index.isin(train_data.index)]
# train_data.to_csv("./data/train.csv")
# test_data.to_csv("./data/test.csv")
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class color2grad_dataset(Dataset):
    def __init__(self, csv):
        self.data = pd.read_csv(csv) # Set to True in case of out of memory
    def __len__(self):
        return self.data['X'].count()

    def __getitem__(self,idx):
        X = self.data['X'][idx]
        Y = self.data['Y'][idx]
        B = self.data['B'][idx]
        G = self.data['G'][idx]
        R = self.data['R'][idx]
        Gx = self.data['Gx'][idx]
        Gy = self.data['Gy'][idx]

        
        return torch.tensor((X,Y,B,G,R),dtype=torch.float32),torch.tensor((Gx,Gy),dtype=torch.float32)




class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)
        self.predict = nn.Linear(hidden_size,output_size)
        self.init_()

    def init_(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.predict.weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.predict(x)

        return x

train_set = color2grad_dataset("./data/train.csv")

train_loader = DataLoader(train_set,batch_size=35000, shuffle=True)

input_size = 5 # X, Y, B, G, R
hidden_size = 20
output_size = 2
dropout_rate = 0.3

# model = MLP(input_size,hidden_size,output_size,dropout_rate).to(device)
model = torch.load("./best.ckpt").to(device)
model.train()

learning_rate = 0.001
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


num_epochs = 500

best_loss = 10000

loss_record = []
total_step = len(train_loader)
for epoch in tqdm(range(501, 501+num_epochs)):
    for i, (data, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        data = data.to(device)
        labels = labels.to(device)

        outputs = model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
            loss_record.append(loss.item())

        if loss.item() < best_loss:
            print("Best Epoch: {}".format(epoch))
            best_loss = loss.item()
            torch.save(model,"best.ckpt")
    # print('Epoch [{}/{}], Loss: {:.4f}'
    #               .format(epoch + 1, num_epochs, loss.item()))

torch.save(model, 'threelayernd.ckpt')
loss_record = np.array(loss_record)
np.savetxt("./data/loss_record.txt", loss_record)