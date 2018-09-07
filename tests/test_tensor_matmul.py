import unittest
import pytest

import numpy as np

from autograd.tensor import Tensor

class TestTensorMatMul(unittest.TestCase):
    def test_simple_matmul(self):
        # t1 is (3, 2)
        t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)

        # t2 is a (2, 1)
        t2 = Tensor([[10], [20]], requires_grad=True)

        t3 = t1 @ t2

        assert t3.data.tolist() == [[50], [110], [170]]

        grad = Tensor([[-1], [-2], [-3]])
        t3.backward(grad)

        np.testing.assert_array_equal(t1.grad.data,
                                      grad.data @ t2.data.T)

        np.testing.assert_array_equal(t2.grad.data,
                                      t1.data.T @ grad.data)
