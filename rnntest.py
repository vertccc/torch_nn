import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv(r'C:\Users\Xu\Desktop\if09.csv')
    # 1 min 取点
    df = df[::120]
    # feature ma
    df['ma'] = df.lastpx.rolling(10).mean()
    # 10 min 后的涨跌幅度
    df['y'] = df.lastpx.shift(-10)/df.lastpx - 1
    df = df.dropna()
    # x,y 按 batch 和 timestep长度 reshape
    timestep,input_size = 20,7
    batch_n = len(df)//timestep
    L = timestep * batch_n
    xall = df[['bidpx','bidvol','askpx','askvol','lastpx','volume','ma']][:L].values
    yall = df['y'][:L].values 
    # x batch timestep inputsize
    x_np = xall.reshape(batch_n,timestep,input_size)
    # y batch timestep 1
    yall = yall[timestep-1::timestep]
    y_np = yall.reshape(batch_n,1,1)
    # y_np = yall.reshape(batch_n,timestep,1)
    return x_np,y_np


class MyRNN(nn.Module):
    def __init__(self):
        input_size = 7
        hidden_size = 12
        num_layers = 2
        output_size = 1
        super(MyRNN,self).__init__()
        # self.input_size = input_size
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        # batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        # bidirectional – If True, becomes a bidirectional(*2) LSTM. Default: False(*1)
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        # 设置初始状态h_0与c_0
        # num_layers*num_directions, batch, hidden_size
        # h0 = Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size))
        # c0 = Variable(torch.zeros(self.num_layers,x.size(0),self.hidden_size))
        # forward x of shape (batch,seq_len,input_size) since batch_first is True
        # out,(h_n,c_n) = self.lstm(x,(h0,c0))
        x,_ = self.lstm(x)
        b,s,h = x.shape
        x = x.contiguous()
        x = x.view(s*b,h)
        x = self.fc(x)
        x = x.view(b,s,-1)
        # out = self.fc(out[:, -1, :])
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


predictor = model.eval()

y_p = predictor(Variable(X).float()).reshape(1,-1).detach().numpy().tolist()[0]
y_x = Y_np.reshape(1,-1)[0]
plt.plot(y_p, 'r', label='prediction')
plt.plot(y_x, 'b', label='real')
plt.legend(loc='best')
