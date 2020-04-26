import torch
import numpy as np
from torch.autograd import Variable
'''
data = np.array([1,2,3])
t = torch.tensor(data)  # on cpu by default, int32
t = torch.Tensor(data) # float
t = t.cuda() # move to gpu
'''

x_data = [1.,2.,3.]
y_data = [2.,4.,6.]

w = Variable(torch.Tensor([1]),requires_grad=True)

def forward(x):
    return x*w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)
    # return abs(y_pred-y)

print('predict (before training)',4,forward(4).data[0])

for epoch in range(100):
    for x_val,y_val in zip(x_data,y_data):
        l = loss(x_val,y_val)
        l.backward()
        w.data = w.data - 0.01*w.grad.data
        # Manually zero the gradients after updating weights
        w.grad.data.zero_()
    print('progress:',epoch,l.data[0])

print('predict (after training)',4,forward(4).data[0])