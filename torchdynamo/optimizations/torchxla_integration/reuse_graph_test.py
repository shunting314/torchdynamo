import unittest
import torch
from torch import fx, nn
import copy
import torchdynamo.optimizations.torchxla_integration as integration

class BasicModule(nn.Module):
    def __init__(self):
        super(BasicModule, self).__init__()

    def forward(self, x, y):
        return x + y

    def get_random_inputs(self):
        return (torch.randn(10), torch.randn(10))

class MatmulModule(nn.Module):
    def __init__(self):
        super(MatmulModule, self).__init__()

    def forward(self, x, y):
        return x @ y

    def get_random_inputs(self):
        return (torch.randn(5, 100), torch.randn(100, 5))

class LinearModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)

    def get_random_inputs(self):
        return (torch.randn(10),)

def allclose(expected, actual):
    def unwrap(cont):
        if isinstance(cont, (list, tuple)) and len(cont) == 1:
            return cont[0]
        return cont
    expected = unwrap(expected)
    actual = unwrap(actual)

    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return torch.allclose(expected, actual)
    elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all(torch.allclose(a, b) for a, b in zip(expected, actual))
    else:
        raise RuntimeError("Unexpected types")

def make_reuse_graph_test(module_class, niter=100):
    def test_wrapper(self):
        print("Enter test case")
        import torch_xla.core.xla_model as xm
        xla_dev = xm.xla_device()
        mod = module_class()
        xla_module = copy.deepcopy(mod).to(device=xla_dev)
        inputs = mod.get_random_inputs()
        optimized_mod = integration.extract_compiled_graph(fx.symbolic_trace(mod), inputs)

        for i in range(niter):
            rand_args = mod.get_random_inputs()
            orig_dev = rand_args[0].device
            rand_args_copy = copy.deepcopy(rand_args)

            # Can not simply call
            #   expected = mod(*rand_args)
            # Since we need use xla to calculate expected results
            xla_inputs = tuple(copy.deepcopy(inp).to(device=xla_dev) for inp in rand_args)
            xla_out = xla_module(*xla_inputs)
            expected = xla_out.to(device=orig_dev)

            actual = optimized_mod(*rand_args_copy)

            if not allclose(expected, actual):
                print(f"Incorrect results at iter {i}. expected\n{expected}, actual\n{actual}")
                self.assertTrue(False)

            # make sure arguments match after calling the model forward method
            # to handle inplace updates.
            if not allclose(rand_args, rand_args_copy):
                print(f"Incorrect updated arguments at iter {i}. expected\n{rand_args}, actual\n{rand_args_copy}")
                self.asesrtTrue(False)
    return test_wrapper

class TorchXLAReuseGraphTest(unittest.TestCase):
    test_basic = make_reuse_graph_test(BasicModule)
    test_matmul = make_reuse_graph_test(MatmulModule)
    test_linear = make_reuse_graph_test(LinearModule)
