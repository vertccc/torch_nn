import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_train = np.random.randn(100,4)
x_train = x_train.astype('float32')
w0 = np.array([[1],[1],[2],[2]],dtype=np.float32)
y_train = x_train.dot(w0)


df = pd.read_csv(r'C:\Users\Xu\Desktop\fpfp.csv')
df['cons'] = [1000]*len(df)
x_train = np.array(df[['fpif','fpih']],dtype='float32')
y_train =np.array(df.fpic,dtype='float32').reshape(len(df),1)



num_epochs = 600000
learning_rate = 1e-6



#x = torch.from_numpy(x_train)
#y =  torch.from_numpy(y_train)
#w = torch.randn(2, 1, device=torch.device("cpu"), dtype=torch.float, requires_grad=True)
#print(w)
#for t in range(num_epochs):
#    y_pred = x.mm(w)
#    loss = (y_pred - y).pow(2).sum()
#    if loss.item() != loss.item():
#        print(t,w)
#        break
##    print('loss',t, loss.item()) 
#    loss.backward()
#    with torch.no_grad():
##        print('grad',w.grad)
#        w -= learning_rate * w.grad 
#        w.grad.zero_()
##        print('new w',w)
#print(w)



# Linear regression model  2. 定义网络结构 y=w*x+b 其中w的size [1,1], b的size[1,]
#model = nn.Linear(2,1,False)
#
## Loss and optimizer 3.定义损失函数， 使用的是最小平方误差函数
#criterion = nn.MSELoss()
### 4.定义迭代优化算法， 使用的是随机梯度下降算法
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  
#loss_dict = []
### Train the model 5. 迭代训练
##
##
#for epoch in range(num_epochs):
#    # Convert numpy arrays to torch tensors  5.1 准备tensor的训练数据和标签
#    inputs = torch.from_numpy(x_train)
#    targets = torch.from_numpy(y_train)
#
#    # Forward pass  5.2 前向传播计算网络结构的输出结果
#    outputs = model(inputs)
#    # 5.3 计算损失函数
#    loss = criterion(outputs, targets)
#    
#    # Backward and optimize 5.4 反向传播更新参数
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
##
##    
##    # 可选 5.5 打印训练信息和保存loss
#    loss_dict.append(loss.item())
#    if (epoch+1) % 5 == 0:
#        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
##print(model.state_dict())
        
        
        
        
        
# Plot the graph 画出原y与x的曲线与网络结构拟合后的曲线
#predicted = model(torch.from_numpy(x_train)).detach().numpy()
#plt.plot(x_train, y_train, 'ro', label='Original data')
#plt.plot(x_train, predicted, label='Fitted line')
#plt.legend()
#plt.show()
#
## 画loss在迭代过程中的变化情况
#plt.plot(loss_dict, label='loss for every epoch')
#plt.legend()
#plt.show()
