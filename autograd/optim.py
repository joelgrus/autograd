"""
Optimizers go here
"""
from autograd.module import Module

class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def step(self, module: Module) -> None:
        for parameter in module.parameters():
            parameter -= parameter.grad * self.lr
