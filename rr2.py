import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np

def load_data():
    df = pd.read_csv(r'C:\Users\Xu\Desktop\if09.csv')
    # 1 min 取点
    df = df[::120]
    # feature ma
    df['ma'] = df.lastpx.rolling(10).mean()
    # 10 min 后的涨跌幅度
    df['y'] = df.lastpx.shift(-10)/df.lastpx - 1
    df = df.dropna()
    
    xall = df[['bidpx','bidvol','askpx','askvol','lastpx','volume','ma']].values
    yall = df['y'].values 
    
    x_np = xall.reshape(-1,1,7)
    # y batch 1 1
    y_np = yall.reshape(-1,1,1)
    return x_np,y_np


class MyRNN(nn.Module):
    def __init__(self):
        input_size = 7
        hidden_size = 12
        num_layers = 2
        output_size = 1
        super(MyRNN,self).__init__()
        self.lstmx = nn.LSTM(input_size,hidden_size,num_layers)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
#        x = x.float()
        x,_ = self.lstmx(x)
        s, b, h = x.shape
        x = x.view(s*b, h) # view改变维度；转换成线性层的输入格式，因为nn.Linear 不接受三维的输入，所以我们先将前两维合并在一起
        x = self.fc(x) # 经过线性层
        x = x.view(s, b, -1) # 将前两维分开，最后输出结果
        return x

# data prepars
# X batch*timestep*inputsize
# X_np = [[[1,3,8,3],[4,5,6,4]],[[2,2,1,4],[1,45,1,10]],[[1,2,1,4],[9,9,2,1]]]
        
X_np,Y_np = load_data()
X = torch.from_numpy(X_np)
Y = torch.from_numpy(Y_np)



# paras 
learning_rate = 0.1
epochs = 10
# construct model criteria and optimizer
model = MyRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
for epoch in range(epochs):
    var_x = Variable(X).float()
    var_y = Variable(Y).float()
    # forward
    output = model(var_x)
    loss = criterion(output,var_y)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# modelpredict = model.eval()
