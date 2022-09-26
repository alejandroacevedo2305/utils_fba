import warnings 
warnings.filterwarnings("ignore")
from cobra import Model, Reaction, Metabolite
import cobra
import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
import itertools
import copy
from cobra.util.array import create_stoichiometric_matrix
import networkx as nx
from networkx.algorithms import bipartite
import torch
#from custom_clases import GINo
import copy
import networkx as nx
import pickle
from torch.utils.data import RandomSampler
from torch.nn import Linear, LeakyReLU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
from torch_geometric.nn.models import GIN
import numpy as np
from sklearn.model_selection import train_test_split


class regresor_GIN(torch.nn.Module):
    
    def __init__(
        self, 
        target_node_idx: int,
        n_nodes : int, 
        num_features : int, 
        out_channels: int,
        dropout : float = 0.08, 
        hidden_dim : int = 30, 
        #heads : int = 5,
        LeakyReLU_slope : float = 0.01,

        num_layers: int = 3
    ):
        super(regresor_GIN, self).__init__() # TODO: why SUPER gato? 
        self.n_nodes = n_nodes
        self.dropout = dropout
        self.num_features = num_features
        self.target_node_idx = target_node_idx
        self.out_channels = out_channels
        
        self.GIN_layers =  GIN(in_channels= num_features, hidden_channels= hidden_dim, num_layers= num_layers, out_channels= out_channels, dropout=dropout,  jk='lstm', act='LeakyReLU', act_first = True)#.to('cuda')        
        self.FC1        = Linear(in_features=out_channels, out_features=1, bias=True)#.to('cuda')
        #self.FC2        = Linear(in_features=n_nodes, out_features=1, bias=True)#.to('cuda')        #self.leakyrelu = LeakyReLU(LeakyReLU_slope).to('cuda')
        self.leakyrelu = LeakyReLU(LeakyReLU_slope)#.to('cuda')
    def forward(
        self, x, 
        edge_index, # 
        batch_size
    ):

     x     = self.GIN_layers(x, edge_index)
     x     = x.reshape(batch_size, self.n_nodes, self.out_channels)
     x     = self.FC1(self.leakyrelu(x))
        
     return  x[:,self.target_node_idx,:].squeeze()
 
import pandas as pd
import seaborn as sns

def show_results(modelo, loader, batch_size:int=250):

    predictions: list = []
    true_values: list = []
    modelo.eval()
    for data in loader:
        prediction = modelo(data.x, data.edge_index, batch_size)
        predictions.extend(prediction.squeeze().tolist())
        true_values.extend(data.y.squeeze().tolist())

    sorted_idxs = np.argsort(true_values)


    evaluation = pd.DataFrame(
        {'Actual data': [true_values[i] for i in sorted_idxs],
        'Prediction':[predictions[i] for i in sorted_idxs]}
    ) 
    sns.lineplot(data=evaluation)


def make_loader(graphs: list, batch_size: int  =250, num_samples: int = 500):

    sampler_train_set = RandomSampler(
        graphs,
        num_samples= num_samples, #params["training"]["sampler_num_samples"],  # Genera un muestreo del grafo
        replacement=True,  # con repeticion de muestras
    )
    return DataLoader(graphs, batch_size=batch_size, sampler = sampler_train_set,  drop_last=True)



def train(optimizer: torch.optim, loss_fun: torch.nn, modelo: regresor_GIN,loader: DataLoader, batch_size:int = 250):
    modelo.train()
    for data in loader:
        prediction = modelo(data.x, data.edge_index, batch_size)
        loss       = loss_fun(prediction, data.y)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        #print(loss)
    return loss
def evaluate(modelo: regresor_GIN, loss_fun: torch.nn, loader: DataLoader,  total_eval_loss: float = 0, batch_size:int = 250):
    #total_eval_loss: float = 0
    modelo.eval()
    for data in loader:        
        prediction = modelo(data.x, data.edge_index, batch_size)
        loss_eval       = loss_fun(prediction, data.y)
        total_eval_loss += loss_eval.item()
    return total_eval_loss    



def train_and_evaluate(optimizer,loss_fun, modelo, train_loader,test_loader,save_state_dict: bool = False, epochs: int = 100, min_total_loss_val: float= 1e10, verbose: bool = False, batch_size:int = 250 ):
    
    for epoch in range(epochs):
        loss = train(optimizer,loss_fun, modelo,train_loader, batch_size)
        if epoch % 1 == 0:
            total_eval_loss = evaluate(modelo,loss_fun,test_loader, batch_size)       
            if total_eval_loss     < min_total_loss_val:
                min_total_loss_val = total_eval_loss
                best_eval_weights  = copy.deepcopy(modelo.state_dict())
                #best_model =  copy.deepcopy(modelo.state_dict())
                if verbose:
                    print(f"NEW best \t{min_total_loss_val}")
                if save_state_dict:
                    torch.save(best_eval_weights, "state_dict_best_evaluated_model.pth")
                    if verbose:
                        print(f"state_dict_best_evaluated_model.pth overwritten")
    return best_eval_weights




def cargar_metabolismo():
  toy                = Model('toy_metabolism')

  def new_reaction(model: cobra.core.model.Model, identifier: str, name: str, subsystem:str, stoichiometry:str, lower_bound: float =False, upper_bound:float=False):
      rxn               = Reaction(identifier)
      model.add_reaction(rxn)
      rxn.build_reaction_from_string(stoichiometry)
      rxn.subsystem = subsystem
      rxn.name      = name
      if lower_bound:
          rxn.lower_bound = lower_bound  # This is the default
      if upper_bound:
        rxn.upper_bound = upper_bound  # This is the default
  # Respiration     
  new_reaction(toy, 'O2_difussion', '', 'Respiration',                 "O2[e] <--",                       -10, 0)
  new_reaction(toy, 'O2_uptake'   , '', 'Respiration',                 "O2[e] --> O2[c]",                0, 10)
  new_reaction(toy, 'OXPHOS'      , '', 'Respiration',                 "20 NADH[c] + O2[c] --> ATP[c]", 0, 10)
  #new_reaction(toy, 'ATP_sink',     '', 'Respiration',                 "ATP[c] --> ",                   .2, .2)
  #glicolisis
  new_reaction(toy, 'A_diffusion', '', 'Glycolysis', "A[e] <--")
  new_reaction(toy, 'A_uptake', '', 'Glycolysis', "A[e] --> A[c]", 0, 3.3)
  new_reaction(toy, 'R1', '', 'Glycolysis', "ATP[c] + A[c] --> B[c]")
  new_reaction(toy, 'R2', '', 'Glycolysis', "B[c] --> 2 ATP[c] + 2 C0[c] + 2 NADH[c]")
  #Amino Acid
  new_reaction(toy, 'R3',          '', 'Amino_Acids', ".5 C0[c] + 2 NADH[c] <--> 20 L[c]")
  new_reaction(toy, 'L_exchange',  '', 'Amino_Acids', "L[e] <--> L[c]", -10, 10)
  new_reaction(toy, 'L_diffusion', '', 'Amino_Acids', "L[e] <-->", -10, 10)
  #TCA cycle
  new_reaction(toy, 'DH1', '', 'TCA_cycle', "C0[c] --> 0.7 C1[c] + NADH[c]")
  new_reaction(toy, 'DH2', '', 'TCA_cycle', "C1[c] --> 0.3 C2[c] + NADH[c]")
  new_reaction(toy, 'DH3', '', 'TCA_cycle', "C2[c] --> C0[c]     + NADH[c]")
  #Biomass
  new_reaction(toy, 'Growth_rate', '', 'Growth_rate', "17 ATP[c] + 7 C0[c] + 2 C1[c] + 3 C2[c] --> biomass[c]")
  new_reaction(toy, 'biomass_balance', '', 'Growth_rate', "biomass[c] -->")
  toy.objective = 'Growth_rate'
  #solution = toy.optimize()
  return toy
def cultivar(model, init_biomasa = .07, init_glucosa = 1, init_oxigeno = .02, init_etanol = 0, horas= 4, init_ATP = .1):

  def add_dynamic_bounds(model, y):
      """Use external concentrations to bound the uptake flux of glucose."""
      biomass, glucose, O2, L, ATP, B = y  # expand the boundary species
      glucose_max_import = -.9 * glucose / (.7 + glucose)
      model.reactions.A_diffusion.lower_bound = glucose_max_import
      
      oxygen_max_import = -40 * O2 / (2 + O2)
      model.reactions.O2_difussion.lower_bound = oxygen_max_import
      

      model.reactions.L_diffusion.lower_bound = -5 * L / (1 + L)
      
      
  all_times = []
  all_fluxes = []
  def dynamic_system(t, y):
      """Calculate the time derivative of external species."""

      biomass, glucose, O2, L, ATP, B   = y  # expand the boundary species

      # Calculate the specific exchanges fluxes at the given external concentrations.
      with model:
          add_dynamic_bounds(model, y)

          cobra.util.add_lp_feasibility(model)
          feasibility = cobra.util.fix_objective_as_constraint(model)
          lex_constraints = cobra.util.add_lexicographic_constraints(
              model, ['Growth_rate', 'A_diffusion', 'O2_difussion', 'L_diffusion','OXPHOS', 'R1'], ['max', 'max', 'max', 'max', 'max', 'max'])
          
          all_fluxes.append(lex_constraints.values)
          all_times.append([t])

      # Since the calculated fluxes are specific rates, we multiply them by the
      # biomass concentration to get the bulk exchange rates.
      #print(lex_constraints.values)
      
      fluxes = lex_constraints.values
      fluxes *= biomass


      # This implementation is **not** efficient, so I display the current
      # simulation time using a progress bar.
      if dynamic_system.pbar is not None:
          dynamic_system.pbar.update(1)
          dynamic_system.pbar.set_description('t = {:.3f}'.format(t))
          #print(lex_constraints.values)

      return fluxes

  dynamic_system.pbar = None

  def infeasible_event(t, y):
      """
      Determine solution feasibility.

      Avoiding infeasible solutions is handled by solve_ivp's built-in event detection.
      This function re-solves the LP to determine whether or not the solution is feasible
      (and if not, how far it is from feasibility). When the sign of this function changes
      from -epsilon to positive, we know the solution is no longer feasible.

      """

      with model:

          add_dynamic_bounds(model, y)

          cobra.util.add_lp_feasibility(model)
          feasibility = cobra.util.fix_objective_as_constraint(model)

      return feasibility - infeasible_event.epsilon

  infeasible_event.epsilon = 1E-6
  infeasible_event.direction = 1
  infeasible_event.terminal = True

  ts = np.linspace(0, horas, 400)  # Desired integration resolution and interval
  # ['Biomass', 'A_diffusion', 'O2_difussion', 'L_diffusion','OXPHOS', 'DH1']
  #y0 = [.07,     1.,             0.02,                0.0, 0.1, 0.1]
  
  y0 = [init_biomasa, init_glucosa, init_oxigeno, init_etanol, init_ATP, 0.1]
  

  with tqdm() as pbar:
      dynamic_system.pbar = pbar

      sol = solve_ivp(
          fun=dynamic_system,
          events=[infeasible_event],
          t_span=(ts.min(), ts.max()),
          y0=y0,
          t_eval=ts,
          rtol=1e-6,
          atol=1e-8,
          method='BDF'
      )
  time = sol.t
  biomass, glucose, O2, L, ATP, B = tuple((sol.y))
  
  
  concentraciones =  (biomass, glucose, O2, L, ATP)
  flujos          = (all_fluxes, all_times)
  
  return time, concentraciones, flujos

def graficar(time, A, B, C, D, E, labels ):

  fig, ax = plt.subplots()
  fig.subplots_adjust(right=0.75)

  twin1 = ax.twinx()
  twin2 = ax.twinx()
  twin3 = ax.twinx()
  twin4 = ax.twinx()
  # Offset the right spine of twin2.  The ticks and label have already been
  # placed on the right by twinx above.
  twin2.spines.right.set_position(("axes", 1.2))
  twin3.spines.right.set_position(("axes", 1.4))
  twin4.spines.right.set_position(("axes", 1.6))

  p1, = ax.plot(time, A, "c-", label=labels[0])
  p2, = twin1.plot(time, B, "r-", label=labels[1])
  p3, = twin2.plot(time, C, "b-", label=labels[2])
  p4, = twin3.plot(time, D, "g-", label=labels[3])
  p5, = twin4.plot(time, E, "y-", label=labels[4])
  #ax.set_xlim(0, 7)
  #ax.set_ylim(0.03, .4)
  #twin1.set_ylim(0.039, 0.23)
  #twin2.set_ylim(0.07, .9)
  #twin3.set_ylim(0, 10)

  ax.set_xlabel('tiempo')
  ax.set_ylabel(labels[0])
  twin1.set_ylabel(labels[1])
  twin2.set_ylabel(labels[2])
  twin3.set_ylabel(labels[3])
  twin4.set_ylabel(labels[4])

  ax.yaxis.label.set_color(p1.get_color())
  twin1.yaxis.label.set_color(p2.get_color())
  twin2.yaxis.label.set_color(p3.get_color())
  twin3.yaxis.label.set_color(p4.get_color())
  twin4.yaxis.label.set_color(p5.get_color())

  tkw = dict(size=4, width=1.5)
  ax.tick_params(axis='y', colors=p1.get_color(), **tkw)
  twin1.tick_params(axis='y', colors=p2.get_color(), **tkw)
  twin2.tick_params(axis='y', colors=p3.get_color(), **tkw)
  twin3.tick_params(axis='y', colors=p4.get_color(), **tkw)
  twin4.tick_params(axis='y', colors=p5.get_color(), **tkw)
  ax.tick_params(axis='x', **tkw)

  ax.legend(handles=[p1, p2, p3, p4, p5])

  plt.show()
  

def graficar_experimento(tiempo, concentraciones, flujos):
  
  biomass, glucose, O2, L, ATP = concentraciones
  all_fluxes, all_times = flujos

  Growth_rate, A_diffusion, O2_difussion, L_diffusion,OXPHOS, _ = tuple(np.array(all_fluxes).T.__abs__())
  all_time_iters = np.array(all_times).squeeze()
  idxs = [np.argmin(abs(all_time_iters - t)**2) for t in tiempo]

  collapsed_Biomass = [Growth_rate[i] for i in idxs]
  collapsed_A_diffusion = [A_diffusion[i] for i in idxs]
  collapsed_O2_difussion = [O2_difussion[i] for i in idxs]
  collapsed_L_diffusion = [L_diffusion[i] for i in idxs]
  collapsed_OXPHOS = [OXPHOS[i] for i in idxs]


  graficar(tiempo, biomass, glucose, O2, L, ATP, ['biomass', 'glucose', 'oxygen', 'ethanol', 'ATP' ])
  graficar(tiempo, collapsed_Biomass, collapsed_A_diffusion, collapsed_O2_difussion, collapsed_L_diffusion, collapsed_OXPHOS, ['growth rate', 'glucose uptake', 'oxygen uptake', 'ethanol production', 'OxPhos' ])


def cobra_a_networkx(model):
  S_matrix = create_stoichiometric_matrix(model)

  n_mets, n_rxns = S_matrix.shape
  assert (
      n_rxns > n_mets
  ), f"Usualmente tienes mas metabolitos ({n_mets}) que reacciones ({n_rxns})"

  # Constructo raro de matrices cero
  # fmt: off
  S_projected = np.vstack(
      (
          np.hstack( # 0_mets, S_matrix
              (np.zeros((n_mets, n_mets)), S_matrix)
          ),  
          np.hstack( # S_trns, 0_rxns
              (S_matrix.T * -1, np.zeros((n_rxns, n_rxns)))
          ),
      )
  )
  S_projected_directionality = np.sign(S_projected).astype(int)
  G = nx.from_numpy_matrix(
      S_projected_directionality, 
      create_using=nx.DiGraph, 
      parallel_edges=False
  )

  # Cosas sorprendentemente no cursed
  # fmt: off
  #formulas: list[str] = [recon2.reactions[i].reaction for i in range(recon2.reactions.__len__())]
  rxn_list: list[str] = [model.reactions[i].id       for i in range(model.reactions.__len__())]
  met_list: list[str] = [model.metabolites[i].id     for i in range(model.metabolites.__len__())]

  assert n_rxns == rxn_list.__len__()
  assert n_mets == met_list.__len__()

  node_list : list[str] = met_list + rxn_list 
  part_list : list[dict[str, int]] = [{"bipartite": 0} for _ in range(n_rxns)] + [{"bipartite": 1} for _ in range(n_mets)]

  nx.set_node_attributes(G, dict(enumerate(part_list)))
  G = nx.relabel_nodes(G, dict(enumerate(node_list)))
  assert G.is_directed() 


  largest_wcc = max(nx.connected_components(nx.Graph(G)), key=len)


  # Create a subgraph SG based on G
  SG = G.__class__()
  SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)


  SG.add_edges_from((n, nbr, d)
      for n, nbrs in G.adj.items() if n in largest_wcc
      for nbr, d in nbrs.items() if nbr in largest_wcc)

  SG.graph.update(G.graph)

  assert G.nodes.__len__() >= SG.nodes.__len__()
  assert G.edges.__len__() >= SG.edges.__len__()
  assert SG.nodes.__len__() == len(largest_wcc)
  assert SG.is_directed() 
  assert nx.is_connected(nx.Graph(SG))

  grafo_nx                     = SG
  rxn_partition , met_partition,   = bipartite.sets(grafo_nx)
  #return  grafo_nx
  #assert set(rxn_list) == rxn_partition and set(met_list) == met_partition
  return  grafo_nx, node_list

def guardar_experimento(model,tiempo, concentraciones, flujos):
    grafo_nx, node_list = cobra_a_networkx(model)
    biomass, glucose, O2, L, ATP = concentraciones
    all_fluxes, all_times = flujos
    
    Growth_rate, A_diffusion, O2_difussion, L_diffusion,OXPHOS, _ = tuple(np.array(all_fluxes).T.__abs__())
    all_time_iters = np.array(all_times).squeeze()
    idxs = [np.argmin(abs(all_time_iters - t)**2) for t in tiempo]

    collapsed_Biomass = [Growth_rate[i] for i in idxs]
    collapsed_A_diffusion = [A_diffusion[i] for i in idxs]
    collapsed_O2_difussion = [O2_difussion[i] for i in idxs]
    collapsed_L_diffusion = [L_diffusion[i] for i in idxs]
    collapsed_OXPHOS = [OXPHOS[i] for i in idxs]

    

    #nodes_features = dict(zip(grafo_nx.nodes, node_list.__len__()*np.zeros((1,glucose.tolist().__len__())).tolist()))
    nodes_features  = dict(zip(grafo_nx.nodes, itertools.repeat([])))
    new_features = {'Growth_rate': np.around(collapsed_Biomass, 7).tolist(),
    'A_diffusion': np.around(collapsed_A_diffusion, 7).tolist(),
    'O2_difussion': np.around(collapsed_O2_difussion, 7).tolist(),
    'L_diffusion': np.around(collapsed_L_diffusion, 7).tolist(),
   'OXPHOS': np.around(collapsed_OXPHOS, 7).tolist(),
   'biomass[c]': np.around(biomass, 7).tolist(),
    'A[e]': np.around(glucose, 7).tolist(),
    'O2[e]': np.around(O2, 7).tolist(),
    'L[e]': np.around(L, 7).tolist(),
   'ATP[c]': np.around(ATP, 7).tolist()}
    
    
    
    
    nodes_features.update(new_features)
    

    assert nodes_features.__len__() == grafo_nx.nodes.__len__()

    nx.set_node_attributes(grafo_nx, nodes_features, "x")


    #nx.write_gpickle(grafo_nx, "experimento.gpickle")
  
    return grafo_nx

def agregar_dos_experimentos(grafo_nx_1, grafo_nx_2):
    


    def grow_dict(previo, nuevo):  
        """Esto es un .update(overwritte=False)?"""
        updated_dict =  {**previo}
        not_new_keys = set(updated_dict.keys()).intersection(set(nuevo.keys()))
        new_keys     = set(nuevo.keys()).difference(set(updated_dict.keys()))
        
        
        for new in new_keys:
            updated_dict[new] =np.array(nuevo[new]).tolist() 
        for common in not_new_keys:
            updated_dict[common].extend(np.array(nuevo[common]).tolist() )
        return updated_dict

    attrs_1    = copy.deepcopy({**nx.get_node_attributes(grafo_nx_1, "x")})
    attrs_2    = copy.deepcopy({**nx.get_node_attributes(grafo_nx_2, "x")})

    
    
    
    updated_attr  = grow_dict(attrs_1, attrs_2)
    updated_graph = copy.deepcopy(grafo_nx_1)
    

    nx.set_node_attributes(updated_graph, updated_attr, "x")

    return updated_graph

from torch_geometric.utils.convert import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
from torch_geometric.nn.models import GIN
import numpy as np
from sklearn.model_selection import train_test_split


device = torch.device("cpu")
if torch.cuda.is_available():
    torch.cuda.init()
    assert torch.cuda.is_initialized(), "CUDA no iniciado!"
    device = torch.device("cuda:0")
print(device)

def crear_lista_de_grafos(experimentos, window_length: int=6):
    nx_G         = copy.deepcopy(experimentos)
    x_attribute     = nx.get_node_attributes(nx_G, "x")
    longest_feature = max(len(v) for k,v in x_attribute.items())
    equal_len_attrs = {k:(longest_feature*[0] if len(v) == 0 else v) for k,v in x_attribute.items()}
    nx.set_node_attributes(nx_G, equal_len_attrs, 'x')
    assert len(set(len(v) for k,v in nx.get_node_attributes(nx_G, "x").items())) == 1
    pyg_graph       = from_networkx(nx.Graph(nx_G))#nx_G)#nx_G)#nx.Graph(nx_G))
    #dict(zip(my_dict.keys(), map(f, my_dict.values())))
    assert pyg_graph.x.shape[1]  == pyg_graph.num_features == longest_feature

    producto_idx = list(nx_G.nodes()).index( 'L[e]')
    producto_features = nx_G.nodes()['L[e]']['x']


    assert set(pyg_graph.x[producto_idx,:].numpy()).__len__() ==  set(np.array(producto_features)).__len__()
    assert np.allclose(pyg_graph.x[producto_idx,:].numpy()[0:producto_features.__len__()], np.array(producto_features))


    def sliding_window(ts, features,producto_idx, target_len = 1):
        X, Y = [], []
        for i in range(features + target_len, ts.shape[1] + 1):  #en este caso sería de 14+1 hasta el final de la serie 
            
            
            X.append(ts[:,i - (features + target_len):i - target_len]) #15 - 15, posicion 0 : 15-1 = 14 valores
            Y.append(ts[producto_idx, i - target_len:i]) #15-1 = 14 : 15  1 valor [14,15]
            
        return X, Y
    #window_length = 1
    X, y =sliding_window(pyg_graph.x, window_length, producto_idx, target_len = 1)
    lista_de_grafos = []

    for graph_x, target in  copy.deepcopy(zip(X, y)):
        graph_x[producto_idx,:] = 0
        
        nuevo = copy.deepcopy(pyg_graph)
        nuevo.x = torch.tensor(graph_x).float()
        nuevo.y =  torch.tensor(target).float()
        lista_de_grafos.append(nuevo.to(device))
    return lista_de_grafos, pyg_graph, producto_idx, window_length


def entrenar_red_neuronal(experimentos, epochs:int=1000, batch_size: int=250, window_length: int = 6):


    lista_de_grafos_1, pyg_graph, producto_idx, _ = crear_lista_de_grafos(experimentos[0],window_length)
    lista_de_grafos_2, pyg_graph, producto_idx, _ = crear_lista_de_grafos(experimentos[1],window_length)
    lista_de_grafos_3, pyg_graph, producto_idx, _ = crear_lista_de_grafos(experimentos[2],window_length)
    lista_de_grafos_4, pyg_graph, producto_idx, _ = crear_lista_de_grafos(experimentos[3],window_length)
    lista_de_grafos_5, pyg_graph, producto_idx, _ = crear_lista_de_grafos(experimentos[4],window_length)
    lista_de_grafos_6, pyg_graph, producto_idx, _ = crear_lista_de_grafos(experimentos[5],window_length)

    lista_de_grafos = []

    for l in [lista_de_grafos_1, lista_de_grafos_2, lista_de_grafos_3, lista_de_grafos_4, lista_de_grafos_5, lista_de_grafos_6]:
        lista_de_grafos.extend(l)
        
    Train_graphs, Test_graphs = train_test_split(
            lista_de_grafos,test_size=0.25,shuffle=True)

    #batch_size   = 250
    out_channels = 20
    epochs       = 1000
    n_nodes      = pyg_graph.num_nodes
    train_loader = make_loader(Train_graphs, batch_size)
    test_loader  = make_loader(Test_graphs, batch_size)
    modelo = regresor_GIN(target_node_idx = producto_idx,n_nodes = n_nodes, num_features = window_length, out_channels = out_channels, hidden_dim =20).to(device)    
        #modelo.load_state_dict(torch.load("state_dict_best_evaluated_model.pth"))
    loss_fun  = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(
    modelo.parameters(),
            lr= 0.0017681039331400813,  # Learning rate
            weight_decay= 0.0067108394238172536,  # Decaimiento de pesos
        )
    best_eval_weights = train_and_evaluate(optimizer,loss_fun, modelo, train_loader,test_loader, epochs, batch_size=batch_size)
    modelo.load_state_dict(best_eval_weights)
    show_results(modelo, train_loader, batch_size)
    return copy.deepcopy(modelo)

def probar_red_neuronal_entrenada(red_neuronal_entrenada, PRUEBA):    
    lista_de_grafos, _, _, _ = crear_lista_de_grafos(PRUEBA,6)
    batch_size   = 250
    loader       = make_loader(lista_de_grafos, batch_size)
    show_results(red_neuronal_entrenada.eval(), loader, batch_size)
    
    
    
def OLD_entrenar_red_neuronal(experimentos, epochs:int=200, batch_size: int=250):
    
    lista_de_grafos, pyg_graph, producto_idx, window_length = crear_lista_de_grafos(experimentos)


    Train_graphs, Test_graphs = train_test_split(
        lista_de_grafos,test_size=0.35,shuffle=True)
    #[i.y for i in lista_de_grafos[-5:]]
    #loader = DataLoader(lista_de_grafos, batch_size=3, shuffle=False)    
    #[l.y for l in list(loader)]

    #batch_size   = 250
    out_channels = 30
    n_nodes      = pyg_graph.num_nodes
    train_loader = make_loader(Train_graphs, batch_size)
    test_loader  = make_loader(Test_graphs, batch_size)
    modelo = regresor_GIN(target_node_idx = producto_idx,n_nodes = n_nodes, num_features = window_length, out_channels = out_channels).to(device)    
    #modelo.load_state_dict(torch.load("state_dict_best_evaluated_model.pth"))
    loss_fun  = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(
        modelo.parameters(),
        lr= 0.0017681039331400813,  # Learning rate
        weight_decay= 0.0067108394238172536,  # Decaimiento de pesos
    )
    best_eval_weights = train_and_evaluate(optimizer,loss_fun, modelo, train_loader,test_loader, epochs, batch_size=batch_size)
    modelo.load_state_dict(best_eval_weights)
    show_results(modelo, test_loader, batch_size)
    
    return copy.deepcopy(modelo)
