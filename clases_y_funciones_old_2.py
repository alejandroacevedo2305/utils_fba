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
import gc




def ver_metabolismo(metabolismo):

    df =pd.DataFrame({
    "Nombre": [metabolismo.reactions[r].id for r in range(len(metabolismo.reactions))],
    "Descripción": [metabolismo.reactions[r].name for r in range(len(metabolismo.reactions))],
    "Reacción": [metabolismo.reactions[r].reaction for r in range(len(metabolismo.reactions))],
    "Ruta": [metabolismo.reactions[r].subsystem for r in range(len(metabolismo.reactions))]})
    
    
    
    
    return df[(df['Nombre'] != 'O2_supply') &  (df['Nombre'] != 'Glc_supply') &  (df['Nombre'] != 'BiomassBalance')&  (df['Nombre'] != 'ETOH_production')]




class regresor_GIN(torch.nn.Module):
    
    def __init__(
        self, 
        target_node_idx: int,
        n_nodes : int, 
        num_features : int, 
        out_channels: int,
        dropout : float = 0.09, 
        hidden_dim : int = 5, 
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
        {'Datos de los experimentos': [true_values[i] for i in sorted_idxs],
        'Predicciones de la red neuronal':[predictions[i] for i in sorted_idxs]}
    ) 
    sns.lineplot(data=evaluation)


def make_loader(graphs: list, batch_size: int  =250, num_samples: int = 800):

    sampler_train_set = RandomSampler(
        graphs,
        num_samples= num_samples, #params["training"]["sampler_num_samples"],  # Genera un muestreo del grafo
        replacement=True,  # con repeticion de muestras
    )
    return DataLoader(graphs, batch_size=batch_size, sampler = sampler_train_set,  drop_last=True)


def train(optimizer: torch.optim, loss_fun: torch.nn, modelo: regresor_GIN,loader: DataLoader, batch_size:int = 250):
    #modelo.to('cuda')
    check_seen_y = []
    modelo.train()
    for data in loader:
        
        with torch.cuda.device('cuda'):
            
             modelo.to('cuda')            
             data.to('cuda')
             prediction = modelo(data.x, data.edge_index, batch_size = batch_size)
             loss       = loss_fun(prediction, data.y)
             check_seen_y.extend(data.y.squeeze().tolist())
             loss.backward()  # Derive gradients.
             optimizer.step()  # Update parameters based on gradients.
             optimizer.zero_grad()  # Clear gradients.
        #print(loss)
    return check_seen_y
def evaluate(modelo: regresor_GIN, loss_fun: torch.nn, loader: DataLoader,  total_eval_loss: float = 0, batch_size:int = 250):
    #total_eval_loss: float = 0
    #modelo.to('cuda')

    modelo.eval()
    for data in loader:
        with torch.cuda.device('cuda'):
             modelo.to('cuda')            
             data.to('cuda')
             prediction = modelo(data.x, data.edge_index, batch_size = batch_size)
             loss_eval       = loss_fun(prediction, data.y)
             total_eval_loss += loss_eval.item()
    return total_eval_loss    
from tqdm import tqdm
from torch.optim.lr_scheduler import *
def train_and_evaluate(optimizer,loss_fun, modelo, train_loader,test_loader,save_state_dict: bool = False, epochs: int = 100, min_total_loss_val: float= 1e10, verbose: bool = False, batch_size:int = 250 ):
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    for epoch in tqdm(range(epochs)):
        gc.collect()
        torch.cuda.empty_cache()          
        total_eval_loss: float = 0
        check_seen_y = train(optimizer,loss_fun, modelo,train_loader, batch_size=batch_size)
        
        if epoch % 1 == 0:
            total_eval_loss = evaluate(modelo,loss_fun,test_loader,total_eval_loss=total_eval_loss ,batch_size=batch_size)
            scheduler.step(total_eval_loss)       
            if total_eval_loss     < min_total_loss_val:
                min_total_loss_val = total_eval_loss
                best_eval_weights  = copy.deepcopy(modelo.state_dict())
                #best_model =  copy.deepcopy(modelo.state_dict())
                if verbose:
                    print(f"NEW best min_total_loss_val {min_total_loss_val} epoch: {epoch}")
                if save_state_dict:
                    torch.save(best_eval_weights, "results/state_dicts/state_dict_best_evaluated_model.pth")
                    if verbose:
                        print(f"state_dict_best_evaluated_model.pth overwritten")
    return best_eval_weights, check_seen_y



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
  new_reaction(toy, 'O2_supply', 'Disponibilidad (oferta) de oxígeno extracelular',  'Respiración', "O2[e] <--      ", -10, 0)
  new_reaction(toy, 'O2_uptake', 'Consumo (demanda) de oxígeno',                     'Respiración', "O2[e] --> O2[c]",  0, 10)
  new_reaction(toy, 'OxPhos',    'Fosforilacion oxidativa',                          'Respiración', "20 NADH[c] + O2[c] --> ATP[c]", 0, 10)
  #new_reaction(toy, 'ATP_sink',     '', 'Respiration',                 "ATP[c] --> ",                   .2, .2)
  #glicolisis
  new_reaction(toy, 'Glc_supply', 'Disponibilidad (oferta) de glucosa extracelular', 'Glicólisis', "Glc[e] <--")
  new_reaction(toy, 'Glc_uptake', 'Consumo de glucosa',                              'Glicólisis', "Glc[e] --> Glc[c]", 0, 10)
  new_reaction(toy, 'G1', 'Consumo de ATP en glicólisis',                            'Glicólisis', "ATP[c] + Glc[c] --> B[c]")
  new_reaction(toy, 'G2', 'Producción de ATP y NADH en glicólisis',                  'Glicólisis', "B[c] --> 2 ATP[c] + 2 Pyr[c] + 2 NADH[c]")
  #Amino Acid
  new_reaction(toy, 'ADH','Síntesis de etanol',                                        'Fermentacion', ".5 Pyr[c] + 2 NADH[c] <--> 20 Etoh[c]")
  new_reaction(toy, 'ETOH_exchange',  'Transporte de etanol entre citoplasma y medio', 'Fermentacion', "Etoh[e] <--> Etoh[c]", -10, 10)
  new_reaction(toy, 'ETOH_production', 'Difusión  de etanol en medio extracelular',      'Fermentacion', "Etoh[e] <-->", -10, 10)
  #TCA cycle
  new_reaction(toy, 'DH1', 'Primera deshidrogenasa del ciclo', 'Ciclo de Krebs', "Pyr[c] --> 0.7 C1[c] + NADH[c]")
  new_reaction(toy, 'DH2', 'Segunda deshidrogenasa del ciclo', 'Ciclo de Krebs', "C1[c] --> 0.3 C2[c] + NADH[c]")
  new_reaction(toy, 'DH3', 'Tercera deshidrogenasa del ciclo', 'Ciclo de Krebs', "C2[c] --> Pyr[c]     + NADH[c]")
  #Biomass
  new_reaction(toy, 'GrowthRate', 'Velocidad específica (tasa) de crecimiento', "Producción de biomasa" ,"17 ATP[c] + 7 Pyr[c] + 2 C1[c] + 3 C2[c] -->  Biomasa")
  new_reaction(toy, 'BiomassBalance', 'Pseudo-transporte de biomasa para balancear', "Producción de biomasa", "Biomasa -->")
  toy.objective = 'GrowthRate'
  #solution = toy.optimize()
  return toy
def cultivar(model, init_biomasa = .07, init_glucosa = 1, init_oxigeno = .02, init_etanol = 0, horas= 4, init_ATP = .1):

  def add_dynamic_bounds(model, y):
      """Use external concentrations to bound the uptake flux of glucose."""
      biomass, glucose, O2, L, ATP, B = y  # expand the boundary species

      model.reactions.Glc_supply.lower_bound = -10 * glucose / (.4 + glucose)

      model.reactions.O2_supply.lower_bound = -3 * O2 / (2 + O2)
      

      model.reactions.ETOH_production.lower_bound = -1 * L / (5 + L)
      
      model.reactions.GrowthRate.upper_bound = 1 * biomass / (1 + biomass)
      
      #model.reactions.OXPHOS.upper_bound = 1.7 * ATP / (.1 + ATP)
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
              model, ['GrowthRate', 'Glc_supply', 'O2_supply', 'ETOH_production','OxPhos', 'G1'], ['max', 'max', 'min', 'max', 'max', 'max'])
          
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
  biomass, glucose, O2, L, ATP, _ = tuple((sol.y))
  
  
  concentraciones =  (biomass, glucose, O2, L, ATP)
  flujos          = (all_fluxes, all_times)
  graficar_experimento(time,concentraciones, flujos)
  return time, concentraciones, flujos

def graficar(time, A, B, C, D, E, labels,title):

  fig, ax = plt.subplots(figsize=(8, 2.5))
  fig.subplots_adjust(right=0.75)

  twin1 = ax.twinx()
  twin2 = ax.twinx()
  twin3 = ax.twinx()
  twin4 = ax.twinx()
  # Offset the right spine of twin2.  The ticks and label have already been
  # placed on the right by twinx above.
  
  twin2.spines.right.set_position(("axes", 1.2))
  twin3.spines.right.set_position(("axes", 1.45))
  twin4.spines.right.set_position(("axes", 1.65))

  p1, = ax.plot(time, A, "g-", label=labels[0], linewidth=3, alpha=0.55)
  p2, = twin1.plot(time, B, "r-", label=labels[1], linewidth=3, alpha=0.55)
  p3, = twin2.plot(time, C, "b-", label=labels[2], linewidth=3, alpha=0.55)
  p4, = twin3.plot(time, D, "c-", label=labels[3], linewidth=3, alpha=0.55)
  p5, = twin4.plot(time, E, "y-", label=labels[4], linewidth=3, alpha=0.55)
  #ax.set_xlim(0, .07)
  
 
  ax.set_ylim(   0.9*min(A), 1.14*max(A))
  
  twin4.set_ylim(0.9*min(E), 1.04*max(E))
     
  
  if max(B) < 0:
      #twin1.invert_yaxis()
      twin1.set_ylim(0.9*max(B), 1.6*min(B))
      
  else:
      twin1.set_ylim(0.9*min(B), 1.1*max(B))
      
  if max(C) < 0:
      #twin2.invert_yaxis()
      twin2.set_ylim(.9*max(C),   1.3*min(C))
      
  else:
      twin2.set_ylim(   0.9*min(C), 1.1*max(C))
  

  #twin2.set_ylim(0.9*min(C), 1.2*max(C))
  
  if max(D) < 0.0001:
      twin3.set_ylim(0.9*min(D), 1e4*max(D))
  else:
      twin3.set_ylim(0.9*min(D), 1.4*max(D))
  
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

  ax.legend(handles=[p1, p2, p3, p4, p5],loc='center left', bbox_to_anchor=(1.8, 0.5))
  plt.title(title)
  plt.show()
  #return ax
  

def graficar_experimento(tiempo, concentraciones, flujos):
  
  biomass, glucose, O2, L, ATP = concentraciones
  all_fluxes, all_times = flujos

  Growth_rate, A_diffusion, O2_difussion, L_diffusion,OXPHOS, _ = tuple(np.array(all_fluxes).T)#.__abs__())
  all_time_iters = np.array(all_times).squeeze()
  idxs = [np.argmin(abs(all_time_iters - t)**2) for t in tiempo]

  collapsed_Biomass = [Growth_rate[i] for i in idxs]
  collapsed_A_diffusion = [A_diffusion[i] for i in idxs]
  collapsed_O2_difussion = [O2_difussion[i] for i in idxs]
  collapsed_L_diffusion = [L_diffusion[i] for i in idxs]
  collapsed_OXPHOS = [OXPHOS[i] for i in idxs]


  graficar(tiempo, biomass, glucose, O2, L, ATP, ['Concentración biomasa', 'Concentracion de Glucosa (Glc[e])',
                                                  'Concentracion de Oxigeno (O2[e])', 'Concentracion de Etanol (Etoh[e])', 
                                                  'Concentracion de ATP[c]' ], title = 'Concentraciones')
  #graficar(tiempo, collapsed_Biomass, collapsed_A_diffusion, collapsed_O2_difussion, collapsed_L_diffusion, collapsed_OXPHOS,
  #         ['Tasa de crecimiento (GrowthRate)',
  #          'Consumo de glucosa (Glc_uptake)', 'Consumo de oxigeno (O2_uptake)',
  #          'Producción de etanol (ADH)', 'Fosforilacion oxidativa (OxPhos)' ], title = 'Flujos')


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
    
    Growth_rate, A_diffusion, O2_difussion, L_diffusion,OXPHOS, _ = tuple(np.array(all_fluxes).T)#.__abs__())
    all_time_iters = np.array(all_times).squeeze()
    idxs = [np.argmin(abs(all_time_iters - t)**2) for t in tiempo]

    collapsed_Biomass = [Growth_rate[i] for i in idxs]
    collapsed_A_diffusion = [A_diffusion[i] for i in idxs]
    collapsed_O2_difussion = [O2_difussion[i] for i in idxs]
    collapsed_L_diffusion = [L_diffusion[i] for i in idxs]
    collapsed_OXPHOS = [OXPHOS[i] for i in idxs]

    

    #nodes_features = dict(zip(grafo_nx.nodes, node_list.__len__()*np.zeros((1,glucose.tolist().__len__())).tolist()))
    nodes_features  = dict(zip(grafo_nx.nodes, itertools.repeat([])))
    new_features = {
    'GrowthRate': np.around(collapsed_Biomass, 7).tolist(),
    'Glc_supply': np.around(collapsed_A_diffusion, 7).tolist(),
    'O2_supply': np.around(collapsed_O2_difussion, 7).tolist(),
    'ETOH_production': np.around(collapsed_L_diffusion, 7).tolist(),
   'OxPhos': np.around(collapsed_OXPHOS, 7).tolist(),
   'Biomasa': np.around(biomass, 7).tolist(),
    'Glc[e]': np.around(glucose, 7).tolist(),
    'O2[e]': np.around(O2, 7).tolist(),
    'Etoh[e]': np.around(L, 7).tolist(),
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

def crear_lista_de_grafos(experimentos, window_length: int=3):
    
    nx_G         = copy.deepcopy(experimentos)
    x_attribute     = nx.get_node_attributes(nx_G, "x")
    longest_feature = max(len(v) for k,v in x_attribute.items())
    equal_len_attrs = {k:(longest_feature*[0] if len(v) == 0 else v) for k,v in x_attribute.items()}
    nx.set_node_attributes(nx_G, equal_len_attrs, 'x')
    assert len(set(len(v) for k,v in nx.get_node_attributes(nx_G, "x").items())) == 1
    pyg_graph       = from_networkx(nx.Graph(nx_G))#nx_G)#nx_G)#nx.Graph(nx_G))
    #dict(zip(my_dict.keys(), map(f, my_dict.values())))
    assert pyg_graph.x.shape[1]  == pyg_graph.num_features == longest_feature

    producto_idx = list(nx_G.nodes()).index( 'Etoh[e]')
    producto_features = nx_G.nodes()['Etoh[e]']['x']


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


def entrenar_red_neuronal(experimentos, epochs:int=1000, batch_size: int=250, window_length: int = 3, use_pretrained:bool=False):


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
            lista_de_grafos,test_size=0.3,shuffle=True)

    #batch_size   = 250
    out_channels = 5
    #epochs       = 1000
    n_nodes      = pyg_graph.num_nodes
    train_loader = make_loader(Train_graphs, batch_size=batch_size)
    test_loader  = make_loader(Test_graphs, batch_size=batch_size)
    modelo = regresor_GIN(target_node_idx = producto_idx,n_nodes = n_nodes, num_features = window_length, out_channels = out_channels, hidden_dim =20).to(device)    
    if use_pretrained:
        modelo.load_state_dict(torch.load("red_neuronal_PRE_entrenada.pth"))
    loss_fun  = torch.nn.MSELoss()
    optimizer = torch.optim.RMSprop(
    modelo.parameters(),
            lr= 0.0017681039331400813,  # Learning rate
            weight_decay= 0.0067108394238172536,  # Decaimiento de pesos
        )
    best_eval_weights, check_seen_y = train_and_evaluate(optimizer,loss_fun, modelo, train_loader,test_loader, epochs = epochs, batch_size=batch_size, save_state_dict=False)
    modelo.load_state_dict(best_eval_weights)
    show_results(modelo, train_loader, batch_size)
    return copy.deepcopy(modelo)#, check_seen_y

def probar_red_neuronal_entrenada(red_neuronal_entrenada, PRUEBA, window_length: int = 3):    
    lista_de_grafos, _, _, _ = crear_lista_de_grafos(PRUEBA,window_length)
    batch_size   = 250
    loader       = make_loader(lista_de_grafos, batch_size)
    show_results(red_neuronal_entrenada.eval(), loader, batch_size)
    
    
 
