import torch

a = torch.Tensor([[3, 2, 9],
                  [12, 13, 5],
                  [12, 9, 8]])
b = torch.Tensor([[0, 1, 1],
                  [1, 1, 0],
                  [0, 0, 1]])

print(a*b)

tup = [(912,11),(310,31221)]

print(tup[0][0])