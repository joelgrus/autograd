"""
The idea here is that we'd like to use our library
to minimize a function, say x ** 2
"""
from autograd.tensor import Tensor, mul

x = Tensor([10, -10, 10, -5, 6, 3, 1], requires_grad=True)

# we want to minimize the sum of squares
for i in range(100):
    sum_of_squares = mul(x, x).sum()  # is a 0-tensor
    sum_of_squares.backward()

    # ugly b/c we haven't implemented the stuff yet
    delta_x = mul(Tensor(0.1), x.grad)
    x = Tensor(x.data - delta_x.data, requires_grad=True)

    print(i, sum_of_squares)

