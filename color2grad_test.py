import torch.nn as nn
import numpy as np
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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

test_set = color2grad_dataset("./data/test.csv")

test_loader = DataLoader(test_set,batch_size=35000, shuffle=True)
model = torch.load("./best.ckpt").to(device)
model.eval()
criterion = nn.MSELoss()

avg_loss = 0.0
cnt = 0
with torch.no_grad():
    for idx, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        labels = labels.to(device)
        outputs = model(data)
        loss = criterion(outputs, labels)
        avg_loss += loss.item()
        cnt += 1
    avg_loss = avg_loss / cnt
    # test_tensor = torch.tensor([24,64,67,80,166],dtype=torch.float32).to(device)
    # print(model(test_tensor))
print(avg_loss)

