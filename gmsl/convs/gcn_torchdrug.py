import os
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch.utils import cpp_extension
from torch import distributed as dist
from torch_geometric.nn import GraphConv
class GraphConv(nn.Module):
    """
    Graph convolution operator from `Semi-Supervised Classification with Graph Convolutional Networks`_.

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/pdf/1609.02907.pdf

    Parameters:
        input_dim (int): input dimension
        output_dim (int): output dimension
        edge_input_dim (int, optional): dimension of edge features
        batch_norm (bool, optional): apply batch normalization on nodes or not
        activation (str or function, optional): activation function
    """

    def __init__(self, input_dim, output_dim, edge_input_dim=None, batch_norm=False, activation="relu", **kwargs):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.linear = nn.Linear(input_dim, output_dim)
        if edge_input_dim:
            self.edge_linear = nn.Linear(edge_input_dim, input_dim)
        else:
            self.edge_linear = None

    def message(self, graph, input):
        # add self loop
        node_in = torch.cat([graph.edge_list[:, 0], torch.arange(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in.unsqueeze(-1) + 1
        message = input[node_in]
        if self.edge_linear:
            edge_input = self.edge_linear(graph.edge_feature.float())
            edge_input = torch.cat([edge_input, torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            message += edge_input
        message /= (degree_in[node_in].sqrt() + 1e-10)
        return message

    def aggregate(self, graph, message):
        # add self loop
        node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weights, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1) + 1
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        update = update / (degree_out.sqrt() + 1e-10)
        return update

    def message_and_aggregate(self, graph, input):
        node_in, node_out = graph.edge_list.t()[:2]
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=graph.device)])
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weights, torch.ones(graph.num_node, device=graph.device)])
        degree_in = graph.degree_in + 1
        degree_out = graph.degree_out + 1
        try:
            edge_weight = edge_weight / ((degree_in[node_in] * degree_out[node_out]).sqrt() + 1e-10)

            adjacency = torch.sparse_coo_tensor(torch.stack([node_in, node_out]), edge_weight,
                                                (graph.num_node, graph.num_node))
            update = torch.sparse.mm(adjacency.t(), input)
        except:
            print("!!!")
        if self.edge_linear:
            edge_input = graph.edge_feature.float()
            edge_input = torch.cat([self.edge_linear(edge_input), torch.zeros(graph.num_node, self.input_dim, device=graph.device)])
            edge_weight = edge_weight.unsqueeze(-1)
            node_out = torch.cat([graph.edge_list[:, 1], torch.arange(graph.num_node, device=graph.device)])
            edge_update = scatter_add(edge_input * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            update += edge_update

        return update
    def combine(self, update):
        output = self.linear(update)
        if self.batch_norm:
            output = self.batch_norm(output)
        if self.activation:
            output = self.activation(output)
        return output

    def forward(self, graph, input):
        update = self.message_and_aggregate(graph, input)
        output = self.combine(update)
        return output



def sparse_coo_tensor(indices, values, size):
    """
    Construct a sparse COO tensor without index check. Much faster than `torch.sparse_coo_tensor`_.

    .. _torch.sparse_coo_tensor:
        https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html

    Parameters:
        indices (Tensor): 2D indices of shape (2, n)
        values (Tensor): values of shape (n,)
        size (list): size of the tensor
    """
    return torch_ext.sparse_coo_tensor_unsafe(indices, values, size)

def get_rank():
    """
    Get the rank of this process in distributed processes.

    Return 0 for single process case.
    """
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0
class cached_property(property):
    """
    Cache the property once computed.
    """

    def __init__(self, func):
        self.func = func
        self.__doc__ = func.__doc__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        result = self.func(obj)
        obj.__dict__[self.func.__name__] = result
        return result
class LazyExtensionLoader(object):

    def __init__(self, name, sources, extra_cflags=None, extra_cuda_cflags=None, extra_ldflags=None,
                 extra_include_paths=None, build_directory=None, verbose=False, **kwargs):
        self.name = name
        self.sources = sources
        self.extra_cflags = extra_cflags
        self.extra_cuda_cflags = extra_cuda_cflags
        self.extra_ldflags = extra_ldflags
        self.extra_include_paths = extra_include_paths
        worker_name = "%s_%d" % (name, get_rank())
        self.build_directory = build_directory or cpp_extension._get_build_directory(worker_name, verbose)
        self.verbose = verbose
        self.kwargs = kwargs

    def __getattr__(self, key):
        return getattr(self.module, key)

    @cached_property
    def module(self):
        return cpp_extension.load(self.name, self.sources, self.extra_cflags, self.extra_cuda_cflags,
                                  self.extra_ldflags, self.extra_include_paths, self.build_directory,
                                  self.verbose, **self.kwargs)


def load_extension(name, sources, extra_cflags=None, extra_cuda_cflags=None, **kwargs):
    """
    Load a PyTorch C++ extension just-in-time (JIT).
    Automatically decide the compilation flags if not specified.

    This function performs lazy evaluation and is multi-process-safe.

    See `torch.utils.cpp_extension.load`_ for more details.

    .. _torch.utils.cpp_extension.load:
        https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load
    """
    if extra_cflags is None:
        extra_cflags = ["-Ofast"]
        if torch.backends.openmp.is_available():
            extra_cflags += ["-fopenmp", "-DAT_PARALLEL_OPENMP"]
        else:
            extra_cflags.append("-DAT_PARALLEL_NATIVE")
    if extra_cuda_cflags is None:
        if torch.cuda.is_available():
            extra_cuda_cflags = ["-O3"]
            extra_cflags.append("-DCUDA_OP")
        else:
            new_sources = []
            for source in sources:
                if not cpp_extension._is_cuda_file(source):
                    new_sources.append(source)
            sources = new_sources

    return LazyExtensionLoader(name, sources, extra_cflags, extra_cuda_cflags, **kwargs)



path = os.path.join(os.path.dirname(__file__), "extension")

torch_ext = load_extension("torch_ext", [os.path.join(path, "torch_ext.cpp")])