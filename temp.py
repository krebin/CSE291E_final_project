import torch

x = torch.Tensor([[[1,2,3],[0,5,6]],[[-1,-2,0],[-4,-5,0]],[[1,2,3],[0,5,6]],[[-1,-2,0],[-4,-5,0]]])
y = torch.tensor([[0,0],[1,0],[1,1],[1,1]]).bool()
print(x.size())
x = x.transpose(0,1)
print(x.size())
print(y.size())
