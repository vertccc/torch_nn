import torch
from torch.autograd import Variable



# 1   Design model using class with Variables
x_data = Variable(torch.Tensor([[1],[2],[3]]))
y_data = Variable(torch.Tensor([[2],[4],[6]]))
class Model(torch.nn.Module):
    def __init__(self):
        # in the constructor we instantiate two nn.Linear module
        super(Model,self).__init__()
        self.linear = torch.nn.Linear(1,1) # one in and one out

    def forward(self,x):
        y_pred = self.linear(x)
        return y_pred
model = Model()


# 2   Construct loss and optimizer
'''
Construct our loss function and an Optimizer
The call to model.parameters() in the SGD constructor
will contain the learnable parameters of the two nn.Linear 
modules which are members of the model
'''
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

# 3   Training loop
for epoch in range(500):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_data)
    # Compute loss
    loss = criterion(y_pred,y_data)
    print(epoch,loss.data)
    # zero gradients, perform a backward pass and update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# after training
x_test = Variable(torch.Tensor([[4.0]]))
y_test_pred = model.forward(x_test).data[0][0]
print('predict',4,y_test_pred)