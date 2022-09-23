# DEFINE CLASES PROPIAS PARA LOS MODELOS

from torch.nn import Linear, LeakyReLU
from torch.nn import Module as torch_nn_Module # Posible choque de namespace
from torch_geometric.nn import GATConv # Graph ATention Convolutive network
from torch.nn import Linear, LeakyReLU
from torch.nn import Module as torch_nn_Module # Posible choque de namespace
from torch_geometric.nn import GATConv # Graph ATention Convolutive network
import torch

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList
from typing import Optional, Tuple, Union
from torch_geometric.nn.conv import GATv2Conv,MessagePassing
from torch_geometric.nn.models.basic_gnn import BasicGNN

class GAT_aa(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int or tuple): Size of each input sample, or :obj:`-1` to
            derive the size from the first input(s) to the forward method.
            A tuple corresponds to the sizes of source and target
            dimensionalities.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`None`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode. If specified, the model
            will additionally apply a final linear transformation to transform
            node embeddings to the expected output feature dimensionality.
            (:obj:`None`, :obj:`"last"`, :obj:`"cat"`, :obj:`"max"`,
            :obj:`"lstm"`). (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """
    supports_edge_weight = False
    supports_edge_attr = True

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout, **kwargs)
from torch_geometric.nn.models.basic_gnn import GIN
from torch.nn import Linear
class GINo(torch.nn.Module):
    
    def __init__(
        self, 
        target_node_idx: int,
        n_nodes : int, 
        num_features : int, 
        dropout : float = 0.2, 
        hidden_dim : int = 5, 
        heads : int = 5,
        #LeakyReLU_slope : float = 0.005,
        out_channels: int = 5,
        num_layers: int = 3
    ):
        super(GINo, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.num_features = num_features
        self.target_node_idx = target_node_idx
        self.out_channels = out_channels
        
        self.GIN_layer =  GIN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, out_channels= out_channels, dropout=dropout,  jk='lstm')#.to('cuda')        
        self.FC1        = Linear(in_features=out_channels, out_features=1, bias=True)#.to('cuda')
        self.FC2        = Linear(in_features=n_nodes, out_features=1, bias=True)#.to('cuda')        #self.leakyrelu = LeakyReLU(LeakyReLU_slope).to('cuda')
        self.leakyrelu = LeakyReLU()#.to('cuda')
    def forward(
        self, x, 
        edge_index, # <- ESTA ES LA COSA QUE HACE QUE SEA GEOMETRICO (espacio topologico)
        data = None
    ):

        x = self.GIN_layer(x, edge_index)
        #x = self.leakyrelu(x).float().reshape(1,self.n_nodes) # Colapsamos todo para RALU
        #x = x[self.target_node_idx]
        
        #x = self.FC(x)
        
        return self.FC2(self.leakyrelu(self.FC1(x).squeeze()))

class GATo_v2Conv(torch.nn.Module):
    
    def __init__(
        self, 
        target_node_idx: int,
        n_nodes : int, 
        num_features : int, 
        dropout : float = 0.2, 
        hidden_dim : int = 5, 
        heads : int = 5,
        #LeakyReLU_slope : float = 0.005,
        out_channels: int = 5,
        num_layers: int = 3
    ):
        super(GATo_v2Conv, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.num_features = num_features
        self.target_node_idx = target_node_idx
        self.out_channels = out_channels
        
        self.GAT_layer = GAT_aa(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, out_channels= out_channels, heads = heads, dropout=dropout, v2 = True, jk='lstm', bias=True, act='leakyrelu')#.to('cuda')        
        self.FC1        = Linear(in_features=out_channels, out_features=1, bias=True)#.to('cuda')
        self.FC2        = Linear(in_features=n_nodes, out_features=1, bias=True)#.to('cuda')        #self.leakyrelu = LeakyReLU(LeakyReLU_slope).to('cuda')
        self.leakyrelu = LeakyReLU()#.to('cuda')
    def forward(
        self, x, 
        edge_index, # <- ESTA ES LA COSA QUE HACE QUE SEA GEOMETRICO (espacio topologico)
        data = None
    ):

        x = self.GAT_layer(x, edge_index)
        #x = self.leakyrelu(x).float().reshape(1,self.n_nodes) # Colapsamos todo para RALU
        #x = x[self.target_node_idx]
        
        #x = self.FC(x)
        
        return self.FC2(self.leakyrelu(self.FC1(x).squeeze()))

class GATo_2(torch.nn.Module):
    
    def __init__(
        self, 
        target_node_idx: int,
        n_nodes : int, 
        num_features : int, 
        dropout : float = 0.2, 
        hidden_dim : int = 5, 
        heads : int = 5,
        LeakyReLU_slope : float = 0.005,
        #target_node_idx: int = 0,
    ):
        super(GATo_2, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.num_features = num_features
        self.target_node_idx = target_node_idx
        
        # Capas del modelo
        #fmt: off
        self.input_layer   = GATConv(in_channels=num_features, out_channels=hidden_dim, heads=heads, dropout=dropout)        
        self.hidden_layer_1   = GATConv(in_channels=hidden_dim*heads, out_channels=hidden_dim, heads=heads, dropout=dropout)
        self.hidden_layer_2   = GATConv(in_channels=hidden_dim*heads, out_channels=hidden_dim, heads=heads, dropout=dropout)  
        self.hidden_layer_3   = GATConv(in_channels=hidden_dim*heads, out_channels=hidden_dim, heads=heads, dropout=dropout)  
        self.hidden_layer_4   = GATConv(in_channels=hidden_dim*heads, out_channels=hidden_dim, heads=heads, dropout=dropout)  
        self.output_layer   = GATConv(in_channels=hidden_dim*heads, out_channels=1, heads=1, dropout=dropout)       
        self.FC        = Linear(in_features=n_nodes, out_features=1, bias=True)
        self.leakyrelu = LeakyReLU(LeakyReLU_slope)
    def forward(
        self, x, 
        edge_index, # <- ESTA ES LA COSA QUE HACE QUE SEA GEOMETRICO (espacio topologico)
        data = None
    ):
        x = self.input_layer(x, edge_index)
        x = self.leakyrelu(x) # Reduce la diemnsionalidad (1,a,b) -> (a,b)
        x = self.hidden_layer_1(x, edge_index)
        x = self.leakyrelu(x) 
        x = self.hidden_layer_2(x, edge_index) #x = self.leakyrelu(x) # Reduce la diemnsionalidad (1,a,b) -> (a,b)
        x = self.leakyrelu(x) 
        x = self.hidden_layer_3(x, edge_index)
        x = self.leakyrelu(x) # Reduce la diemnsionalidad (1,a,b) -> (a,b)
        x = self.hidden_layer_4(x, edge_index)
        x = self.leakyrelu(x) 
        x = self.output_layer(x, edge_index).reshape(1,self.n_nodes) 
        x = self.leakyrelu(x).float().reshape(1,self.n_nodes) # Colapsamos todo para RALU
        #x = x.squeeze()[self.target_node_idx]
        
        x = self.FC(x)
        
        return x

class GATo(torch_nn_Module):
    
    def __init__(
        self, 
        n_nodes : int, 
        num_features : int, 
        dropout : float = 0.1, 
        hidden_dim : int = 5, 
        heads : int = 5,
        LeakyReLU_slope : float = 0.01,
    ):
        super(GATo, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.num_features = num_features
        
        # Capas del modelo
        #fmt: off
        self.layer_1   = GATConv(in_channels=num_features, out_channels=hidden_dim, heads=heads, dropout=dropout)
        self.layer_2   = GATConv(in_channels=hidden_dim*heads, out_channels=1, heads=1, dropout=dropout)
        self.FC        = Linear(in_features=n_nodes, out_features=1, bias=True)
        self.leakyrelu = LeakyReLU(LeakyReLU_slope)
        
    def forward(
        self, x, 
        edge_index, # <- ESTA ES LA COSA QUE HACE QUE SEA GEOMETRICO (espacio topologico)
        data = None
    ):
        x = self.layer_1(x, edge_index)
        x = self.leakyrelu(x.squeeze()) # Reduce la diemnsionalidad (1,a,b) -> (a,b)
        x = self.layer_2(x, edge_index)
        x = self.leakyrelu(x).float().reshape(1,self.n_nodes) # Colapsamos todo para RALU
        x = self.FC(x)
        return x

from torch.utils.data import Dataset

# Extiende la clase dataset para permitir el uso de grafos
class pyg_graph_dataset(Dataset):
    """
    Extension de la clase Datasets para grafos

    #TODO: describir cosas

    Parameters
    ----------
    graph : graph
        Un grafo en el formato de PyTorch Geometric
    target_idx: int
        bjksad
    """
    def __init__(self, graph, target_idx):
        self.graph      = graph      # Grafo completo
        self.target_idx = target_idx # Nodo objetivo
        
    def __len__(self):
        #self.graph.num_features
        return  self.graph.num_features # TODO: Reparar esto
  
    def __getitem__(self, sample_idx):
        
        
        
        x_cloned = self.graph.x[:,sample_idx].clone()
        x_cloned[self.target_idx] = 0
        x = x_cloned.reshape(self.graph.num_nodes,1)
        y = self.graph.x[self.target_idx,sample_idx].reshape(1,1)

        return x, y, self.graph.edge_index
    


#class pyg_graph_dataset(Dataset):
   # """
    #Extension de la clase Datasets para grafos

    ##TODO: describir cosas

    #Parameters
    #----------
    #graph : graph
        #Un grafo en el formato de PyTorch Geometric
    #target_idx: int
        #bjksad
    #"""
    #def __init__(self, graph, target_idx):
    #    self.graph      = graph      # Grafo completo
    #    self.target_idx = target_idx # Nodo objetivo
        
    #def __len__(self):
    #    #self.graph.num_features
    #    return  self.graph.num_features # TODO: Reparar esto
  
    #def __getitem__(self, sample_idx):
    #    x = self.graph.x[:,sample_idx].reshape(self.graph.num_nodes,1)
    #    y = self.graph.x[self.target_idx,sample_idx].reshape(1,1)

    #    return x, y, self.graph.edge_index

