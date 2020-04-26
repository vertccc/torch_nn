import torch
import matplotlib.pyplot as plt

def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.legend(['value'],loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

def plot_image(img,label):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(img[i][0],cmap='gray',interpolation='none')
        plt.title('image with label {}'.format(label[i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def one_hot(label,depth=10):
    out = torch.zeros(label.size(0),depth)
    idx = torch.LongTensor(label).view(-1,1)
    out.scatter_(dim=1,index=idx,value=1)
    return out
