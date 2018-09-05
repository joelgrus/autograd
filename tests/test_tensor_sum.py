import unittest

from autograd.tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_sum(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum()

        t2.backward()

        assert t1.grad.data.tolist() == [1, 1, 1]

    def test_sum_with_grad(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = t1.sum()

        t2.backward(Tensor(3.))

        assert t1.grad.data.tolist() == [3, 3, 3]

