from typing import Iterator
import inspect

from autograd.tensor import Tensor
from autograd.parameter import Parameter

class Module:
    def parameters(self) -> Iterator[Parameter]:
        for name, value in inspect.getmembers(self):
            if isinstance(value, Parameter):
                yield value
            elif isinstance(value, Module):
                yield from value.parameters()

    def zero_grad(self):
        for parameter in self.parameters():
            parameter.zero_grad()
