import torch
from torch.nn import functional as F
from torch import optim
from torch import nn
import time
import torchvision
import matplotlib.pyplot as plt
from mnist_utils import plot_curve,one_hot,plot_image

batch_size = 512  # 一次处理多少张图片
# setp 1 load data
train_dataset = torchvision.datasets.MNIST(r'C:\Users\Xu\Downloads\mnist_data',train=True,download=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]))
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=1500,shuffle=True)


test_dataset = torchvision.datasets.MNIST(r'C:\Users\Xu\Downloads\mnist_data',train=False,download=True,transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
]))
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=1500,shuffle=True)
#x,y = next(iter(train_loader))
#print(x,y)
#plot_image(x,y)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        # xw + b
        self.fc1 = nn.Linear(28*28,256)
        self.fc2 = nn.Linear(256,64)
        self.fc3 = nn.Linear(64,10)

    def forward(self,x):
        # x [b,1,28,28]
        # h1 = relu(xw+b)
        x = F.relu(self.fc1(x))
        # h2 = relu(h1w+b)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
criterion = nn.MSELoss()

net.cuda()
criterion.cuda()
t1 = time.time()
for epoch in range(10):
    for batch_idx,(x,y) in enumerate(train_loader):
        x = x.cuda()
        x = x.view(x.size(0),28*28)
        out = net(x)
        y_onehot = one_hot(y)
        y_onehot = y_onehot.cuda()
        loss = criterion(out,y_onehot)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
#        if batch_idx%10 == 0:
#            print(epoch,batch_idx,loss.item())
t2 = time.time()
print('train time used',(t2-t1)/60)


with torch.no_grad():
    # In test phase, we don't need to compute gradients (for memory efficiency)
    tot_correct = 0
    for x,y in test_loader:
        x = x.cuda()
        y = y.cuda()
#         x = x.view(x.size(0),28*28)
        x = x.reshape(-1,28*28)
        out = net(x)
        pred = out.argmax(dim=1)
        correct = pred.eq(y).sum().float().item()
        tot_correct += correct
    acc = tot_correct / len(test_loader.dataset)
    print('acc',acc)
    t3 = time.time()
    print('test time used',(t3-t2)/60)
    
torch.save(net.state_dict(), 'model.ckpt')

#x,y = next(iter(test_loader))
##x = x.cuda()
#out = net(x.view(x.size(0),28*28))
#pred = out.argmax(dim=1)
##x = x.cpu()
##pred = pred.cpu()
#plot_image(x,pred)
