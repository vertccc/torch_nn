import torch

x = torch.tensor(1.)
a = torch.tensor(1.,requires_grad=True)
b = torch.tensor(2.,requires_grad=True)
c = torch.tensor(3.,requires_grad=True)

y = a**2*x + b*x + c

print('before',a.grad,b.grad,c.grad)
grads = torch.autograd.grad(y,[a,b,c])
print('after',grads[0],grads[1],grads[2])

def step_gradient(b_current,w_current,points,learningRate):
    # y = wx + b  loss = (y-wx-b)^2
    b_grad,w_grad = 0,0
    N = len(points)
    for i in range(N):
        x = points[i,0]
        y = points[i,1]
        b_grad += -(2/N)*(y-((w_current*x)+b_current))
        w_grad += -(2/N)*x*(y-((w_current*x)+b_current))
    new_b = b_current - learningRate*b_grad
    new_w = w_current - learningRate*w_grad
    return new_b,new_w