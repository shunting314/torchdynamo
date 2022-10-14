import torch
import copy
import os
from torch import fx
import torch_xla
import torch_xla.core.xla_model as xm
import dataclasses
from typing import Dict, List, Any
import time

debug = os.environ.get("debug_extract_compiled_graph") is not None

@dataclasses.dataclass
class GraphInputMatcher:
    """
    The GraphInputMatcher class setup the graph inputs for future calls after lazy tracing.
    Specifically, those graph inputs corresponding to method parameters should be replaced with the
    arguments for the current call.

    tensor_id_to_arg_idx maps the tensor id to the parameter index.
    graph_input_tensor_ids, graph_input_ivalues list the tensor_id and ivalue for each of the
    TS/XLA graph inputs.
    """
    tensor_id_to_arg_idx: Dict[int, int]
    graph_input_tensor_ids: List[int]
    # there are 2 categories of graph_input_tensors.
    # Category 1: those whose id are not found in tensor_id_to_arg_idx. These are
    # most likely const tensors and we can get its content from graph_input_tensors
    # Category 2: those whose id are found in tensor_id_to_arg_idx. We should get
    #  the tensor from method arguments
    graph_input_ivalues: List[Any]

    # get the real graph input tensors
    def __call__(self, args):
        real_input = []
        for tensor_id, traced_ivalue in zip(self.graph_input_tensor_ids, self.graph_input_ivalues):
            arg_idx = self.tensor_id_to_arg_idx.get(tensor_id, None)
            if arg_idx is None:
                inp = traced_ivalue
            else:
                inp = args[arg_idx]
            real_input.append(inp)
        return real_input

    def map_to_inputs(self, seq):
        """seq has one item for each graph input. This method does filtering and
           returns those items only corresponding to the original model inputs
        """
        assert len(seq) == len(self.graph_input_tensor_ids)
        ret = [None] * len(self.tensor_id_to_arg_idx)
        idx = 0
        for tensor_id, traced_ivalue in zip(self.graph_input_tensor_ids, self.graph_input_ivalues):
            arg_idx = self.tensor_id_to_arg_idx.get(tensor_id, None)
            if arg_idx is not None:
                ret[arg_idx] = seq[idx]
            idx += 1
        assert not any(x is None for x in ret)
        return ret

def extract_compiled_graph(model: torch.fx.GraphModule, example_inputs):
    orig_device = example_inputs[0].device
    xla_dev = xm.xla_device()
    xla_model = copy.deepcopy(model).to(device=xla_dev)
    xla_args = [arg.to(device=xla_dev) for arg in example_inputs]
    args_tensor_ids = [torch_xla._XLAC._xla_get_tensor_id(xla_arg) for xla_arg in xla_args]

    if debug:
        print(f"args_tensor_ids {args_tensor_ids}")

    tensor_id_to_arg_idx = {tensor_id: i for i, tensor_id in enumerate(args_tensor_ids)}
    xla_out = xla_model(*xla_args)
    args_need_update = torch_xla._XLAC._check_tensor_need_materialization(xla_args)

    if not isinstance(xla_out, (tuple, list)):
        xla_out = (xla_out,)

    args_and_out = tuple(xla_args) + tuple(xla_out)

    if debug or True:
        print(f"XLA IR Text: {torch_xla._XLAC._get_xla_tensors_text(args_and_out)}")
        # print(f"XLA IR HLO: {torch_xla._XLAC._get_xla_tensors_hlo(args_and_out)}")

    # calculate graph hash
    graph_hash = torch_xla._XLAC._get_graph_hash(args_and_out)
    if debug:
        print("graph_hash", graph_hash)

    graph_input_tensor_ids, graph_input_ivalues = torch_xla._XLAC._get_tensors_xla_device_data_node(args_and_out)
    # tensors in graph_input_ivalues are on XLA devices. Move to eager device
    # graph_input_ivalues = [
    #     t.to(device=orig_device) for t in graph_input_ivalues
    # ]
    if debug:
        print(f"graph_input_tensor_ids {graph_input_tensor_ids}")
    assert len(graph_input_tensor_ids) == len(graph_input_ivalues), f"{len(graph_input_tensor_ids)} v.s. {len(graph_input_ivalues)}"
    graph_input_matcher = GraphInputMatcher(tensor_id_to_arg_idx, graph_input_tensor_ids, graph_input_ivalues)

    # sync xla tensors
    torch_xla._XLAC._xla_sync_multi(args_and_out, [])

    # input all cpu tensors
    def optimized_mod(*args):
        enter_ts = time.time();
        if len(args_and_out) == 0:
            return ()

        assert len(args) > 0 # can not handle no args case for now
        eager_device = args[0].device
        graph_input = graph_input_matcher(args)
        start_ts = time.time()
        res, graph_input_handles, graph_output_handles = torch_xla._XLAC._run_cached_graph(graph_hash, graph_input)
        model_input_handles = graph_input_matcher.map_to_inputs(graph_input_handles)
        print(f"graph_input_handles {graph_input_handles}, graph_output_handles {graph_output_handles}, model_input_handles {model_input_handles}")
        print(f"torchxla reuse compiled graph run_cached_graph takes {time.time() - start_ts} seconds") # TODO

        prepare_output_ts = time.time()

        copy_args_ts = time.time()
        assert len(res) == len(args_and_out)
        ncopy = 0
        
        assert len(args) == len(model_input_handles)
        assert len(args) < len(graph_output_handles)

        for i, nede_update in enumerate(args_need_update):
            if nede_update:
                args[i].copy_(res[i])
                ncopy += 1
        
        # for i, arg in enumerate(args):
        #     if model_input_handles[i] != graph_output_handles[i]:
        #         arg.copy_(res[i])
        #         ncopy += 1
        # if False:
        #     for i, arg in enumerate(args):
        #         # only copy those tensors that get inplace updated
        #         if arg is not res[i]:
        #             arg.copy_(res[i])
        #             ncopy += 1

        print(f"Copy {ncopy} args takes {time.time() - copy_args_ts} seconds")

        # need to convert xla tensor back to eager tensor
        copy_res_ts = time.time()
        result = [x.to(device=eager_device) for x in res[len(args):]]
        print(f"Copy results takes {time.time() - copy_res_ts} seconds")

        print(f"prepare output takes {time.time() - prepare_output_ts} seconds")
        print(f"optimized_mod takes {time.time() - enter_ts} seconds overall")

        return result

    return optimized_mod
