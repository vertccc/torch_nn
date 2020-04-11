import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
import itertools
import matplotlib.pyplot as plt

torch.set_grad_enabled(True)   # default ture
torch.set_printoptions(linewidth=120)

# print('torch version:',torch.__version__)
# print('torchvision version:',torchvision.__version__)

def plot_confusion_matrix(cm,classes,normalize=False):
    cmap = plt.cm.Blues
    title = 'Confusion Matrix'
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=45)
    plt.yticks(tick_marks,classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max()/2
    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        plt.text(j,i,format(cm[i,j],fmt),
            horizontalalignment='center',
            color='white' if cm[i,j]>thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def get_num_correct(preds,labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

def get_all_preds(model,loader):
    all_preds = torch.tensor([])
    for batch in loader:
        images,labels = batch
        preds = model(images)
        all_preds = torch.cat((all_preds,preds),dim=0)
    return all_preds

class NetWork(nn.Module):
    def __init__(self):
        super(NetWork,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=12,kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4,out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=60)
        self.out = nn.Linear(in_features=60,out_features=10)
        
    def forward(self,t):
        # hidden conv layer
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t,kernel_size=2,stride=2)
        # hidden conv layer
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t,kernel_size=2,stride=2)
        # hidden linear layer
        t = t.reshape(-1,12*4*4)
        t = F.relu(self.fc1(t))
        # hidden linear layer
        t = F.relu(self.fc2(t))
        # out layer
        return self.out(t)


train_set = torchvision.datasets.FashionMNIST(
    root=r'C:\Users\Xu\Juno\data\FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor()
        ])
    )



network = NetWork()
train_loader = torch.utils.data.DataLoader(train_set,batch_size=100)
optimizer = optim.Adam(network.parameters(),lr=0.01)


print('training ... ')
for epoch in range(5):
    total_loss, total_correct = 0,0
    for batch in train_loader:
        images,labels = batch
        preds = network(images)
        loss = F.cross_entropy(preds,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += get_num_correct(preds,labels)
    pct_correct = total_correct/len(train_set)
    print('epoch: {}, pct_correct: {:.3f}, loss: {:.1f}'.format(epoch,pct_correct,total_loss))
    
with torch.no_grad():    
    prediction_loader = torch.utils.data.DataLoader(train_set,batch_size=10000)
    train_preds = get_all_preds(network,prediction_loader)
preds_correct = get_num_correct(train_preds,train_set.targets)
print('accuracy {:.2f}'.format(preds_correct/len(train_set)))

# build a confusion matrix
# stacked = torch.stack(
#     (train_set.targets, train_preds.argmax(dim=1)),
#     dim=1
#     )
# cmt = torch.zeros(10,10,dtype=torch.int32)
# for p in stacked:
#     tl,pl = p.tolist()
#     cmt[tl,pl] = cmt[tl,pl] + 1
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(train_set.targets,train_preds.argmax(dim=1))

# plot a confusion matrix
names = (
    'T-shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot'
)
plt.figure(figsize=(10,10))
plot_confusion_matrix(cm,names)
