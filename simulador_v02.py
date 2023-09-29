#%%
#from IA_simulador.utils import *
import numpy as np
import random
from itertools import count, islice
import numpy as np
from typing import Generator
import pandas as pd
import sqlite3
from collections import defaultdict
from typing import Dict, List, Tuple, Union
from typing import Dict, List, Union, Optional
import itertools

import warnings
warnings.filterwarnings("ignore")
import random
import random

def generate_integer(min_val:int=1, avg_val:int=3, max_val:int=7, probabilidad_pausas:float=1.0):
    # Validate probabilidad_pausas
    if probabilidad_pausas < 0 or probabilidad_pausas > 1:
        return -1
        # Validate min_val and max_val
    if min_val > max_val:
        return -1
        # Validate avg_val
    if not min_val <= avg_val <= max_val:
        return -1
        # Initialize weights with 1s
    weights = [1] * (max_val - min_val + 1)
        # Calculate the distance of each possible value from avg_val
    distance_from_avg = [abs(avg_val - i) for i in range(min_val, max_val + 1)]
        # Calculate the total distance for normalization
    total_distance = sum(distance_from_avg)
        # Update weights based on distance from avg_val
    if total_distance == 0:
        weights = [1] * len(weights)
    else:
        weights = [(total_distance - d) / total_distance for d in distance_from_avg]
        # Generate a random integer based on weighted probabilities
    generated_integer = random.choices(range(min_val, max_val + 1), weights=weights, k=1)[0]
        # Determine whether to return zero based on probabilidad_pausas
    if random.random() > probabilidad_pausas:
        return 0
    
    return generated_integer


from typing import Dict, List, Any

def extract_first_skills(agenda: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[int]]:
    # Initialize an empty dictionary to store the results.
    result = {}    
    # Loop through each key-value pair in the agenda dictionary.
    for key, value_list in agenda.items():        
        # Check if the list of dictionaries is not empty for the current key.
        if value_list:            
            # Extract the first 'skills' entry from the first dictionary in the list.
            first_skills = value_list[0].get('propiedades', {}).get('skills', [])            
            # Add the key and the first 'skills' entry to the result dictionary.
            result[key] = first_skills
            
    return result

class MisEscritorios:
    
    def __init__(self, skills:Dict[str, List[int]], configuraciones: Dict[str, str], conexiones: Dict[str, bool] = None, niveles_servicio_x_serie:dict =None):
        """_summary_
        Args:
            skills (Dict[str, List[int]]): Series cargadas para cada escritorio.
            configuraciones (Dict[str, str]): Configuraciones de atencion para cada escritorio. 
            conexiones (Dict[str, bool], optional): estado de la conexion de los escritorios. Defaults to None.
        """        
        self.skills                 = skills
        self.configuraciones        = configuraciones
        #anexar las configuraciones de atenciones al dicionario de las skills usando unir_values_en_tupla. 
        #los key-values por ejemplo quedan así 'escritorio_2': ([2, 5, 6, 7, 13, 16], 'RR')    
        
        self.niveles_servicio_x_serie = niveles_servicio_x_serie
        
        self.skills_configuraciones = unir_values_en_tupla(self.skills, self.configuraciones)      
        #iteramos self.skills_configuraciones para armar el dicionario con los escritorios
        self.escritorios            = {key: #escritorio i
                                    {
                                    "skills":series,#series, #series cargadas en el escritorio i
                                    'contador_bloqueo':None, #campo vacío donde se asignará un iterador que cuenta los minutos que el escritorio estará ocupado en servicio 
                                    'minutos_bloqueo':None, #campo vacío donde se asignarán los minutos que se demora la atención 
                                    'estado':'disponible', #si el escritorio está o no disponible para atender   
                                    'configuracion_atencion':config, #configuración del tipo de antención, FIFO, RR, etc.
                                    'pasos_alternancia': None, #objeto que itera en la tabla con pasos de alternancia.
                                    'conexion':None, #campo vacío donde se asignará el estado de la conexión del escritorio (ON/OFF)
                                    'numero_de_atenciones':0, #min 
                                    'numero_pausas':       None,
                                    'numero_desconexiones': None,
                                    'tiempo_actual_disponible':   0, #max
                                    'tiempo_actual_en_atención':  None,
                                    'tiempo_actual_pausa':        None,
                                    'tiempo_actual_desconectado': None,
                                    'contador_tiempo_disponible': iter(count(start=0, step=1)),
                                    'duracion_pausas': (1, 5, 10), #min, avg, max
                                    'probabilidad_pausas':1 , #probabilidad que la pausa ocurra
                                    } for key,(series, config) in self.skills_configuraciones.items()}        
        if not conexiones:
        #     #si no se provee el estado de los conexiones se asumen todas como True (todos conectados):
             conexiones    = {f"{key}": random.choices([True, False], [1, 0])[0] for key in self.escritorios}
        self.escritorios  = actualizar_conexiones(self.escritorios, conexiones)       
        #self.escritorios  = poner_pasos_alternancia(self.escritorios, pasos_alternancia, self.niveles_servicio_x_serie)
        
        
        #Luego separamos los escritorios en conectados (self.escritorios_ON) y desconectados (self.escritorios_OFF)
        #instanciar con todos los escritorio aparagados
        #self.escritorios_ON, self.escritorios_OFF = separar_por_conexion(self.escritorios)
        self.escritorios_OFF = self.escritorios
        
        self.escritorios_ON = {}

        self.nuevos_escritorios_programados       = []
        self.registros_escritorios                = []
    def actualizar_registos_x_escris(self, hora_actual):
        
        self.hora_actual = hora_actual
        self.registros_escritorios.append({k: { 
                            "hora"    :               self.hora_actual, 
                            'conexion':               v['conexion'], 
                            'estado':                 v['estado'] if v['conexion'] else None, 
                            'configuracion_atencion': v['configuracion_atencion']if v['conexion'] else None,
                            'skills':                 v['skills'] if v['conexion'] else None,
                            } for k,v in 
                            {**self.escritorios_ON, **self.escritorios_OFF}.items()})
            
        
    def sumar_nuevo_escritorio(self, skills:List[int], configuracion_atencion:str, conexion:bool, hora=None):
        """_summary_

        Args:
            skills (List[int]): lista con las series a cargar como skills.
            configuracion_atencion (str): tipo de atención.
            conexion (bool): estado ON/OFF
        """       
        # sourcery skip: avoid-builtin-shadow
        #crear nuevo escritorio
        ids_escritorios  = list({**self.escritorios_ON, **self.escritorios_OFF}.keys())
        id               = generar_nuevo_id_escritorio(ids_escritorios)
        nuevo_escritorio = generar_nuevo_escritorio(id, skills, configuracion_atencion , conexion)
        #addicionar el nuevo escritorio a escritorios_ON o escritorios_OFF dependiendo de su conexion
        if nuevo_escritorio[id]['conexion']:
            self.escritorios_ON.update(nuevo_escritorio)
            print(f"nuevo escritorio ON: {nuevo_escritorio}")
        else:
            self.escritorios_OFF.update(nuevo_escritorio)
            print(f"nuevo escritorio OFF: {nuevo_escritorio}")
            
    def eliminar_escritorio(self, id:str):
        """_summary_

        Args:
            id (str): key del escritorio a eliminar
        """
        # eliminar escritorio si la id está en self.escritorios_ON
        if id in self.escritorios_ON:
            removed_escritorio = self.escritorios_ON.pop(id)
            print(f"Escritorio {id, removed_escritorio} eliminado de la lista ON.")
        # eliminar escritorio si la id está en self.escritorios_OFF
        elif id in self.escritorios_OFF:
            removed_escritorio = self.escritorios_OFF.pop(id)
            print(f"Escritorio {id, removed_escritorio} eliminado de la lista OFF.")            
        else:
            # If the ID is not found in either dictionary
            print(f"No se encontró el escritorio {id}.")       
        

    def cambiar_propiedades_escritorio(self, 
                                       escritorio:str, 
                                       skills:List[int]=None, 
                                       configuracion_atencion:str=None, 
                                       conexion:bool=None, 
                                       duracion_pausas: tuple = None, # min_val:int=None, avg_val:int=None, max_val:int=None, 
                                       probabilidad_pausas:float=None) -> None:
        """_summary_
        Modifica las propiedades del escritorio. Si una propiedad entra vacía se ignora. 
        Args:
            escritorio (str): key del escritorio a modificar.
            skills (List[int], optional): Nueva lista de series para cargar como skills. Defaults to None.
            configuracion_atencion (str, optional): Nueva configuracion de atención. Defaults to None.
            conexion (bool, optional): Nuevo estado de conexion. Defaults to None.
        """

        campos = {
                  'skills': skills,
                  'configuracion_atencion': configuracion_atencion,
                  'conexion': conexion,
                  'duracion_pausas': duracion_pausas, #(min_val, avg_val, max_val),
                  'probabilidad_pausas': probabilidad_pausas,
                  }
        #remover propiedades del escritorio que no se modifican
        campos = {k: v for k, v in campos.items() if v is not None}
        #actualizar escritorio
        update_escritorio(escritorio, campos, self.escritorios_OFF, self.escritorios_ON)
        print(f"{campos} de {escritorio} fue modificado.")
        
        
    def actualizar_conexiones(self, conexiones: Dict[str, bool]):
        """_summary_
        Actualiza el estado de las conexiones de los escritorios
        Args:
            conexiones (Dict[str, bool]): Nuevo estado de las conexiones de los escritorios.
        """
        #Actualizar las conexiones y reescribir las variables que guardan los escritorios de acuerdo al estado de las conexiones.
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion(actualizar_conexiones({**self.escritorios_ON, **self.escritorios_OFF}, conexiones))
        #En los escritorios desconectados, resetear los iteradores que cuentan el tiempo de bloqueo y 
        # poner los escritorios en estados disponible, así quedan listo para volver a conectarse. 
        self.escritorios_OFF                      = reset_escritorios_OFF(self.escritorios_OFF)
        
        print(f"conexiones actualizadas. Off: {list(self.escritorios_OFF.keys())} - On: {list(self.escritorios_ON.keys())}")
    
    def iniciar_atencion(self, escritorio, cliente_seleccionado):
        # iterar los escritorios y emisiones
        #for escr_bloq, emi in zip(escritorios_a_bloqueo, emision):
            
            #extraer los minutos que dura la atención asociada a la emision
        minutos_atencion = round((cliente_seleccionado.FH_AteFin - cliente_seleccionado.FH_AteIni).total_seconds()/60)
            #reescribir campos:
            
        self.escritorios_ON[escritorio]['contador_tiempo_atencion'] = iter(islice(count(start=0, step=1), minutos_atencion))#nuevo contador de minutos limitado por n_minutos
        self.escritorios_ON[escritorio]['estado']           = 'atención'#estado bloqueado significa que está atendiendo al cliente.
        self.escritorios_ON[escritorio]['minutos_atencion']  = minutos_atencion#tiempo de atención     
        self.escritorios[escritorio]['numero_de_atenciones'] += 1 #se guarda en self.escritorios para que no se resetee.
        self.escritorios_ON[escritorio]['numero_de_atenciones'] = self.escritorios[escritorio]['numero_de_atenciones'] 
        
        
    def iniciar_pausa(self, escritorio, generador_pausa = generate_integer):
        
        min_val, avg_val, max_val = self.escritorios_ON[escritorio]['duracion_pausas']
        probabilidad_pausas       = self.escritorios_ON[escritorio]['probabilidad_pausas']
        minutos_pausa             = generador_pausa(min_val, avg_val, max_val, probabilidad_pausas)
        
        self.escritorios_ON[escritorio]['contador_tiempo_pausa'] = iter(islice(count(start=0, step=1), minutos_pausa))#nuevo contador de minutos limitado por n_minutos
        self.escritorios_ON[escritorio]['estado']                = 'pausa'#estado bloqueado significa que está atendiendo al cliente.
        self.escritorios_ON[escritorio]['minutos_pausa']         = minutos_pausa#tiempo de atención  
        
    def iniciar_tiempo_disponible(self,escritorio):
        self.escritorios_ON[escritorio]['contador_tiempo_disponible'] = iter(count(start=0, step=1))
        self.escritorios_ON[escritorio]['estado']                     = 'disponible'#estado bloqueado significa que está atendiendo al cliente.
        
    def iterar_escritorios_disponibles(self, escritorios_disponibles: List[str]):
        
        for escri_dispon in escritorios_disponibles:               
            #avanzamos en un minuto el tiempo que lleva disponible.
            tiempo_disponible = next(self.escritorios_ON[escri_dispon]['contador_tiempo_disponible'], None)
            if tiempo_disponible is not None:
            #guardar el tiempo que lleva disponible
                self.escritorios_ON[escri_dispon]['tiempo_actual_disponible'] = tiempo_disponible
            
    def filtrar_x_estado(self, state: str):   
        """
        extrae los escritorios por el estado (disponible o bloqueado)
        """     
        #obtener estados
        self.estados = {escr_i: {'estado': propiedades['estado'], 'configuracion_atencion': 
                        propiedades['configuracion_atencion']} for escr_i, propiedades in self.escritorios_ON.items()} 
        #extraer por disponibilidad    
        if disponibilidad := [
            key for key, value in self.estados.items() if value['estado'] == state
        ]:
            return disponibilidad
        else:
            print(f"No hay escritorio {state}")
            return False
    
    def iterar_escritorios_bloqueados(self, escritorios_bloqueados: List[str]):
        """_summary_

        Args:
            escritorios_bloqueados (List[str]): lista de escritorios bloqueados (en atención o en pausa) para iterar los minutos. 

        Returns:
            _type_: _description_
        """
        # if not self.filtrar_x_disponibilidad('bloqueado'): #esto tiene que considerar en atención y en pausa
        #     print("no hay escritorios bloqueados, h") #no hay escritorios en atención ni en pausa.
        #     return None
        # else:
            
        for escri_bloq in escritorios_bloqueados:
            #ver si está en atención:
            if self.escritorios_ON[escri_bloq]['estado'] == 'atención':
                
                #avanzamos en un minuto el tiempo de atención
                tiempo_atencion = next(self.escritorios_ON[escri_bloq]['contador_tiempo_atencion'], None)
                #si tiene que continuar en atención
                if tiempo_atencion is not None: 
                    #retortar cuantos minutos le quedan en atención.
                    print(f"escritorio {escri_bloq} está en atención, le quedan {(self.escritorios_ON[escri_bloq]['minutos_atencion'])-(tiempo_atencion)} minutos.")
                #si su tiempo de atención terminó, osea llegó a None:
                else:
                    #iniciar pausa 
                    self.iniciar_pausa(escri_bloq)

            #si el escritorio está en pausa:            
            elif self.escritorios_ON[escri_bloq]['estado'] == 'pausa':
                 #iteramos contador_tiempo_pausa:
                 tiempo_pausa = next(self.escritorios_ON[escri_bloq]['contador_tiempo_pausa'], None)
                 if tiempo_pausa is not None: 
                    #si tiene que seguir en pausa
                    #reportar cuanta pausa le queda
                     print(f"escritorio {escri_bloq} está en pausa, le quedan {(self.escritorios_ON[escri_bloq]['minutos_pausa'])-(tiempo_pausa)} minutos.")                 
                 else:
                    #si termina tiempo en pausa pasa a estado disponible
                    self.iniciar_tiempo_disponible(escri_bloq)
                    #self.escritorios_ON[escri_bloq]['estado'] = 'disponible'
                    print(f"escritorio {escri_bloq} quedó disponible")

                    
    def programar_nuevo_escritorio(self, skills, configuracion_atencion,conexion, hora):
        nuevo = {
            'skills': skills, 
            'configuracion_atencion':configuracion_atencion, 
            'conexion':conexion, 
            "hora": hora
            }
        self.nuevos_escritorios_programados.append(nuevo)
        self.nuevos_escritorios_programados = sort_by_time(self.nuevos_escritorios_programados)
        print(f"nuevo escritorio programado: {nuevo}")

        #return self
    def lanzar_escritorios_programados(self, reloj_simulacion):
        
        if self.nuevos_escritorios_programados: 
            
            for propiedades in self.nuevos_escritorios_programados:
            
            
                config_time = pd.Timestamp(f"{reloj_simulacion.date()} {propiedades['hora']}")
                #print(config_time)
                time_diff = abs(config_time - reloj_simulacion)
                #print(time_diff)
                if time_diff <= timedelta(minutes=.5):                    
                    #del propiedades['hora']
                    self.sumar_nuevo_escritorio(**propiedades)                    
                    print(f"nuevo escritorio lanzado a las {propiedades['hora']}")
                    
                    
    def actualizar_conexiones_y_propiedades(self, un_escritorio, tramo, accion):
        
        
        propiedades = tramo['propiedades'] | {'conexion': accion == 'iniciar'}
    
        self.cambiar_propiedades_escritorio(un_escritorio, **propiedades)
        
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion(
            {**self.escritorios_ON, **self.escritorios_OFF}
        )
        
        self.escritorios_ON  = poner_pasos_alternancia(self.escritorios_ON, pasos_alternancia, self.niveles_servicio_x_serie)
        
        
    def aplicar_agenda(self, hora_actual, agenda):
        
        for idEsc, tramos_un_escritorio in agenda.items():
            
            if tramo_idx_tramo := terminar_un_tramo(hora_actual, tramos_un_escritorio):
                tramo     = tramo_idx_tramo[0]
                idx_tramo = tramo_idx_tramo[1]
                print(f"{idEsc} termina tramo (eliminado de agenda): {tramo}")
                self.actualizar_conexiones_y_propiedades(idEsc, tramo, 'terminar')
                del agenda[idEsc][idx_tramo]   
            
            if tramo:=  iniciar_un_tramo(hora_actual, tramos_un_escritorio):
                #se va seguir ejecutando mientras el tramo sea válido
                #poner alguna flag para q no se vuelva a ejecutar
                print(f"{idEsc} inicia tramo: {tramo}")
                self.actualizar_conexiones_y_propiedades(idEsc, tramo, 'iniciar')
def reporte_SLA_x_serie(df, corte_SLA:float=15):
    result_dict = {}
    result_dict['corte_SLA'] = corte_SLA
    # Calculate the total percentage of rows in 'espera' column that are below corte_SLA
    total_rows = len(df)  # Total number of rows in the DataFrame
    total_below_sla = len(df[df['espera'] < corte_SLA])  # Number of rows below corte_SLA in 'espera' column
    total_percentage = (total_below_sla / total_rows) * 100  # Calculate the percentage
    result_dict['total'] = f"{total_percentage:.2f}%"  # Store the percentage as a string in the dictionary   
    # Loop through each unique 'escritorio' to calculate the percentage of rows in 'espera' below corte_SLA
    for escritorio in df['IdSerie'].unique():
        escritorio_df = df[df['IdSerie'] == escritorio]  # Filter rows for the current 'escritorio'
        escritorio_total_rows = len(escritorio_df)  # Total number of rows for the current 'escritorio'
        escritorio_below_sla = len(escritorio_df[escritorio_df['espera'] < corte_SLA])  # Number of rows below corte_SLA for the current 'escritorio'
        escritorio_percentage = (escritorio_below_sla / escritorio_total_rows) * 100  # Calculate the percentage
        result_dict[f'{escritorio}'] = f"{escritorio_percentage:.2f}%"  # Store the percentage as a string in the dictionary
        

    return result_dict         

def nivel_atencion_x_serie(df, SLA_x_serie):
    
    nivel_atencion_serie = {}
    for serie, (_, espera) in SLA_x_serie.items():
        
        df_una_serie          = df[df['IdSerie'] == serie]    
        total_rows            = len(df_una_serie)     
        total_below_sla       = len(df_una_serie[df_una_serie['espera'] < espera])
        if total_rows ==0:
            nivel_atencion = np.NAN
        else:     
            nivel_atencion      = (total_below_sla / total_rows) * 100 
        nivel_atencion_serie = nivel_atencion_serie | {serie: nivel_atencion}
    return nivel_atencion_serie

def iniciar_un_tramo(hora_actual, tramos_un_escritorio):
    for tramo in tramos_un_escritorio:
        inicio  = pd.Timestamp(f"{hora_actual.date()} {tramo['inicio']}")
        #termino = pd.Timestamp(f"{hora_actual.date()} {tramo['termino']}")
        if (hora_actual >= inicio):
            return tramo
    return False

def terminar_un_tramo(hora_actual, tramos_un_escritorio):

    for idx_tramo, tramo in enumerate(tramos_un_escritorio):
        if tramo['termino'] is None:
            return False
        ##inicio  = pd.Timestamp(f"{hora_actual.date()} {tramo['inicio']}")
        termino = pd.Timestamp(f"{hora_actual.date()} {tramo['termino']}")
        if hora_actual > termino:
            return [tramo, idx_tramo]
    return False 


def balancear_carga_escritorios(desk_dict):
    sorted_desks = sorted(desk_dict.keys(), key=lambda x: (desk_dict[x]['numero_de_atenciones'], -desk_dict[x]['tiempo_actual_disponible']))
    
    return sorted_desks

def prioridad_x_serie(niveles_servicio_x_serie, alpha:float=2, beta:float=1):
    pasos = generar_pasos_para_alternancia(niveles_servicio_x_serie, alpha, beta)

    return {row.serie  : row.prioridad for _, row in pasos[~pasos.serie.duplicated()].drop("posicion", axis=1).iterrows()}

def extract_highest_priority_and_earliest_time_row(df, priorities):
    df['Priority'] = df['IdSerie'].map(priorities)
    
    # Sort the DataFrame first based on the Priority and then on the 'FH_Emi' column
    df_sorted = df.sort_values(by=['Priority', 'FH_Emi'], ascending=[True, True])
    
    # Get the row with the minimum priority value (highest priority) and earliest time
    df_highest_priority_and_earliest_time = df_sorted.iloc[[0]]
    
    return df_highest_priority_and_earliest_time.iloc[0,]  


def map_priority_to_steps(input_data: Dict[int, Dict[str, int]]) -> Dict[int, Dict[str, int]]:

    unique_priorities = sorted(set(val['prioridad'] for val in input_data.values()))
    priority_to_steps = {priority: step for priority, step in zip(unique_priorities, reversed(unique_priorities))}
    for key, inner_dict in input_data.items():
        inner_dict['pasos'] = priority_to_steps[inner_dict['prioridad']]
    
    return input_data


def rank_by_magnitude(arr):
    sorted_indices = sorted(enumerate(arr), key=lambda x: x[1], reverse=True)
    rank_arr = [0] * len(arr)
    rank = 1  # Initialize rank
    for i in range(len(sorted_indices)):
        index, value = sorted_indices[i]
        if i > 0 and value == sorted_indices[i - 1][1]:
            pass  # Don't increment the rank
        else:
            rank = i + 1  # Set new rank
        rank_arr[index] = rank  # Store the rank at the original position
    return rank_arr


def create_multiindex_df(data_dict):

    multi_index_list = []
    sorted_data = sorted(data_dict.items(), key=lambda x: x[1]['prioridad'])
    
    priority_groups = {}
    for k, v in sorted_data:
        priority = v['prioridad']
        if priority not in priority_groups:
            priority_groups[priority] = []
        priority_groups[priority].append((k, v))
    
    position = 0  # To enumerate rows
    for priority, items in priority_groups.items():
        
        random.shuffle(items)
        
        for k, v in items:
            pasos = v['pasos']
            
            # Add each entry 'pasos' number of times
            for _ in range(pasos):
                multi_index_list.append((position, priority, k))
                position += 1
                
    multi_index = pd.MultiIndex.from_tuples(
        multi_index_list, 
        names=["posicion", "prioridad", "serie"]
    )
    
    # Initialize the DataFrame
    df = pd.DataFrame(index=multi_index)
    
    return df

def one_cycle_iterator(series, start_pos):
    part_one = series[start_pos+1:]
    part_two = series[:start_pos]
    complete_cycle = pd.concat([part_one, part_two])
    return iter(complete_cycle)


def calcular_prioridad(porcentaje, espera, alpha:float=2, beta:float=1):
    
    return ((porcentaje**alpha)/(espera**beta))

def generar_pasos_para_alternancia(niveles_servicio_x_serie, alpha:float=2, beta:float=1):

    priority_levels     = rank_by_magnitude([calcular_prioridad(porcentaje, espera, alpha, beta) for (porcentaje, espera) in niveles_servicio_x_serie.values()])
    niveles_de_servicio = [{'porcentaje':porcen, "espera":espera, "prioridad" : priori} for (porcen, espera), priori in zip(niveles_servicio_x_serie.values(),priority_levels)]

    return create_multiindex_df(
                    map_priority_to_steps({s: random.choice(niveles_de_servicio) for s in list(niveles_servicio_x_serie.keys())})
                    ).reset_index(drop=False)

class pasos_alternancia():
    def __init__(self, niveles_servicio_x_serie, skills, alpha:float=2, beta:float=1):
        
        self.pasos             = generar_pasos_para_alternancia(niveles_servicio_x_serie, alpha, beta)
        self.pasos             = self.pasos[self.pasos['serie'].isin(skills)].reset_index(drop=True)
        self.pasos['posicion'] = self.pasos.index
        self.iterador_posicion = itertools.cycle(self.pasos.posicion)
                   
    def buscar_cliente(self, fila_filtrada):
        
        self.posicion_actual        = self.pasos.iloc[next(self.iterador_posicion)]
        serie_en_la_posicion_actual = self.posicion_actual.serie

        if not     fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_la_posicion_actual])].empty:
            print(f"serie_en_la_posicion_actual  {self.posicion_actual.serie} coincidió con cliente(s)")
            #de los clientes q coincidieron retornar el que llegó primero
            return fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_la_posicion_actual])].sort_values(by='FH_Emi', ascending=True).iloc[0,]
        else:
            print(f"serie_en_la_posicion_actual no conincidió, buscando en otros pasos")
            start_position                = self.posicion_actual.posicion
            single_cycle_iterator_x_pasos = one_cycle_iterator(self.pasos.serie, start_position)
            
            for serie_en_un_paso in single_cycle_iterator_x_pasos:
                
                if not     fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_un_paso])].empty:
                    print(f"serie {serie_en_un_paso} en otro paso coincidió con cliente")
                    return fila_filtrada[fila_filtrada.IdSerie.isin([serie_en_un_paso])].sort_values(by='FH_Emi', ascending=True).iloc[0,]
            else:
                raise ValueError("Las series del escritorio no coinciden con la serie del cliente. No se puede atender. ESTO NO DE DEBERIA PASAR, EL FILTRO TIENE QUE ESTAR FUERA DEL OBJETO.")
def poner_pasos_alternancia(escritorios: dict, class_to_instantiate, niveles_servicio_x_serie):
    for key, value in escritorios.items():
        # Check if 'configuracion_atencion' is 'Aternancia'
        if value.get('configuracion_atencion') == 'Aternancia':
            # Instantiate the class and assign it to 'pasos_alternancia'
            value['pasos_alternancia'] = class_to_instantiate(niveles_servicio_x_serie, skills = escritorios[key]['skills'])
            
    return escritorios

class forecast():
    def __init__(self, db: str= 'data/mock_db.sqlite'):
        self.db = db
        self.forecast_model = None

    def _obtener_atenciones(self):
        # Ingresa el nombre de la base de datos del cliente
        # Cleaned_ es para referirse a datos limpios
        cliente_db : str = "Cleaned_" + "V24_Banm3_"

        sql_query : str = f"""
        SELECT * FROM {cliente_db}
        WHERE (FH_Emi > '2023-01-01 00:00:00');
        """
        conn = sqlite3.connect(self.db)

        self.df = pd.read_sql( sql_query, con = conn, parse_dates = [3, 4, 5, 6]).astype({
                "IdOficina" : "Int32",
                "IdSerie" : "Int8",
                "IdEsc" : "Int8",
                "FH_Emi" : "datetime64[s]",
                "FH_Llama" : "datetime64[s]",
                "FH_AteIni" : "datetime64[s]",
                "FH_AteFin" : "datetime64[s]",
            })

        conn.close() # Cierra el SQL connection

        return self
    
    def un_mes(self, mes:str=4, col_emisiones:str='FH_Emi'):
        
        self.df = self._obtener_atenciones().df[self.df[col_emisiones].dt.month == mes].reset_index(drop=True)
        return self

    def _dias_validos(self, col_emisiones:str='FH_Emi'):
        return list(sorted(self.df[col_emisiones].dt.day.unique()))


    def un_dia(self, col_emisiones:str='FH_Emi'):
        
        self.df = self.df[self.df[col_emisiones].dt.day == np.random.choice(self._dias_validos()) ].reset_index(drop=True)
        
        return self
    
    def emisiones(self):
        #for _ , self.row in self.df.iterrows():
        for _ , self.row in self.iterrows():
            yield self.row  # Yield each row as a pandas Series

    
    def emisiones_forecast(self, 
        dia : str = '2023-06-15', 
        hora_ini : str = '07:30:00', 
        hora_fin : str = '16:30:00'):
        
        from modelo_forecast import Forecast_Model

        # /DeepenData/.miniconda/envs/EXT-totalpack/lib/python3.8/site-packages/fbprophet/forecaster.py
         # from fbprophet.make_holidays import get_holiday_names #, make_holidays_df

        if not bool(self.forecast_model):
            self.forecast_model = Forecast_Model( 
                data = self.df
            ).train_model()

        self.df_forecast = self.forecast_model.predice_dia(
            dia = dia,
            hora_ini = hora_ini,
            hora_fin = hora_fin,
        )
        
    def emisiones_generador(self):

        from modelo_forecast import Forecast_Model

        self.df_forecast_cuantizado = Forecast_Model.dia_cuantizado( self.df_forecast )

        for _ , self.row in self.df_forecast_cuantizado.iterrows():
            yield self.row  # Yield each row as a pandas Series
          
def obtener_skills(un_dia):   

    skills_defaultdict  = defaultdict(list)  

    #for index, row in un_dia.df.iterrows():
    for index, row in un_dia.iterrows():

        skills_defaultdict[row['IdEsc']].append(row['IdSerie'])
    for key in skills_defaultdict:
        skills_defaultdict[key] = list(set(skills_defaultdict[key]))
        
    skills = dict(skills_defaultdict)   
        
    return {f"escritorio_{k}": v for k, v in skills.items()}#

def filtrar_escritorios_x_estado(data_dict: Dict[int, List[int]], flag_dict: Dict[int, bool]) -> Dict[int, List[int]]:

    return {key: value for key, value in data_dict.items() if flag_dict.get(key, False)}

def unir_values_en_tupla(data_dict: Dict[int, List[int]], label_dict: Dict[int, str]) -> Dict[int, Tuple[List[int], str]]:

    return {key: (value, label_dict.get(key, '')) for key, value in data_dict.items()}

import re

def generar_nuevo_id_escritorio(lst):
   
    # Initialize an empty set to store the integers already in the list
    existing_integers = set()
    
    # Initialize an empty string to store the common prefix
    common_prefix = ""
    
    # Loop through each element in the list
    for element in lst:
        # Use regular expression to extract the prefix and integer part
        # The regular expression consists of two groups:
        # 1. The prefix (captured as \w+)
        # 2. The integer (captured as \d+)
        match = re.match(r'(\w+)_(\d+)', element)
        
        # If a match is found
        if match:
            # Extract the prefix and integer from the regular expression groups
            prefix, integer = match.groups()            
            # Add the integer to the set (converted to int for numerical comparison)
            existing_integers.add(int(integer))            
            # Set the common prefix (assuming all elements have the same prefix)
            common_prefix = prefix    
    # Find the closest positive integer not in the existing set
    new_integer = 1  # Start with 1 as it's the smallest positive integer
    while new_integer in existing_integers:
        new_integer += 1  # Increment and check again
    
    # Create the new element by combining the common prefix and the new integer
    new_element = f"{common_prefix}_{new_integer}_{'added'}"
    
    return new_element
from copy import deepcopy
def actualizar_conexiones(original_dict, update_dict):  

    # Iterate through each key-value pair in the update_dict
    for key, value in update_dict.items():
        # Check if the key exists in the original_dict

        if key in original_dict:
            # Update the 'conexion' field with the new boolean value
            original_dict[key]['conexion'] = value            

    return deepcopy(original_dict)

def separar_por_conexion(original_dict):  

    # Create deep copies of the original dictionary
    true_dict = deepcopy(original_dict)
    false_dict = deepcopy(original_dict)    

    # Create lists of keys to remove
    keys_to_remove_true = [key for key, value in true_dict.items() if value.get('conexion') is not True]
    keys_to_remove_false = [key for key, value in false_dict.items() if value.get('conexion') is not False]    

    # Remove keys from the deep-copied dictionaries
    for key in keys_to_remove_true:
        del true_dict[key]
    for key in keys_to_remove_false:
        del false_dict[key]
    return (true_dict, false_dict)


def reset_escritorios_OFF(desk_dict):
    # Initialize the update dictionary
    update_dict = {'contador_bloqueo': None,
                   'minutos_bloqueo': None,
                   'estado': 'disponible',
                   }    
    # Loop through each key-value pair in the input dictionary
    for desk, info in desk_dict.items():
        # Update the inner dictionary with the update_dict values
        info.update(update_dict)        
    return desk_dict


def reporte_SLA(df, corte_SLA:float=30):

    result_dict = {}
    result_dict['corte_SLA'] = corte_SLA
    # Calculate the total percentage of rows in 'espera' column that are below corte_SLA
    total_rows = len(df)  # Total number of rows in the DataFrame
    total_below_sla = len(df[df['espera'] < corte_SLA])  # Number of rows below corte_SLA in 'espera' column
    total_percentage = (total_below_sla / total_rows) * 100  # Calculate the percentage
    result_dict['total'] = f"{total_percentage:.2f}%"  # Store the percentage as a string in the dictionary   
    # Loop through each unique 'escritorio' to calculate the percentage of rows in 'espera' below corte_SLA
    for escritorio in df['escritorio'].unique():
        escritorio_df = df[df['escritorio'] == escritorio]  # Filter rows for the current 'escritorio'
        escritorio_total_rows = len(escritorio_df)  # Total number of rows for the current 'escritorio'
        escritorio_below_sla = len(escritorio_df[escritorio_df['espera'] < corte_SLA])  # Number of rows below corte_SLA for the current 'escritorio'
        escritorio_percentage = (escritorio_below_sla / escritorio_total_rows) * 100  # Calculate the percentage
        result_dict[f'{escritorio}'] = f"{escritorio_percentage:.2f}%"  # Store the percentage as a string in the dictionary
        

    return result_dict

def remove_last_row(df: pd.DataFrame) -> pd.DataFrame:
    last_index = df.index[-1]
    df.drop(last_index, inplace=True)   
    df.reset_index(drop=True, inplace=True)  
    return df

# Define the function to update attributes of a given 'escritorio'
def update_escritorio(key, attributes, dict1, dict2):
    # Search for the key in dict1
    if key in dict1:
        # Update only the fields that are passed in attributes
        for attr, value in attributes.items():
            if attr in dict1[key]:
                dict1[key][attr] = value
        #return True
    
    # Search for the key in dict2
    elif key in dict2:
        # Update only the fields that are passed in attributes
        for attr, value in attributes.items():
            if attr in dict2[key]:
                dict2[key][attr] = value
        #return True
    
    # If the key is not found in either dictionary
    else:
        print(f"{key} no está en los escritorios")
        
        
def remove_escritorio(key: str, dict1: dict, dict2: dict) -> None:   
    # Check if the key exists in dict1 and remove it if found
    if key in dict1:
        del dict1[key]
        print(f"Removed {key} from the first dictionary.")
        #return

    # Check if the key exists in dict2 and remove it if found
    if key in dict2:
        del dict2[key]
        print(f"Removed {key} from the second dictionary.")
        #return

    # If the key is not found in either dictionary
    print(f"{key} not found in either dictionary.")
#def nuevo_escriorio(escritorios = svisor.escritorios, nuevo_escritorio):   
# nuevo_escritorio = {'skills': [2, 5, 6, 9, 16], 'configuracion_atencion': 'FIFO', 'conexion' : True}

def generar_nuevo_escritorio(id:str, skills:list, configuracion_atencion:str, conexion:bool)->dict:
    nuevos_campos = {
        'skills': skills,
        'configuracion_atencion': configuracion_atencion,
        'conexion': conexion,
                        } | {
                            'contador_bloqueo': None,
                            'minutos_bloqueo': None,
                            'estado': 'disponible',
                        }
    return  {id: nuevos_campos}


def filtrar_fila_por_skills(df, config_dict):
    skills = config_dict.get('skills', [])
    filtered_df = df[df['IdSerie'].isin(skills)]
    return filtered_df #False if filtered_df.empty else filtered_df

def FIFO(df):
    if df is None or df.empty:
        return None   
    min_time = df['FH_Emi'].min()
    earliest_rows = df[df['FH_Emi'] == min_time]
    if len(earliest_rows) > 1:
        selected_row = earliest_rows.sample(n=1)
    else:
        selected_row = earliest_rows   
    return selected_row.iloc[0]

def x_t_de_Espera(df):

    if df is None or df.empty:
        return None    
    max_wait = df['espera'].max()    
    longest_wait_rows = df[df['espera'] == max_wait]    
    if len(longest_wait_rows) > 1:
        selected_row = longest_wait_rows.sample(n=1)
    else:
        selected_row = longest_wait_rows
    
    return selected_row.iloc[0]


# Function definition
def x_Prioridad_de_serie(df, priority_dict):

    if df is None or df.empty:
        return None
    
    # Create lists for each priority level based on the dictionary
    high_priority = [k for k, v in priority_dict.items() if v == 'alta']
    medium_priority = [k for k, v in priority_dict.items() if v == 'media']
    low_priority = [k for k, v in priority_dict.items() if v == 'baja']
    
    # Loop through priority lists in the order of 'alta', 'media', 'baja'
    for priority_list in [high_priority, medium_priority, low_priority]:
        # Filter rows where 'IdSerie' matches the current priority list
        filtered_df = df[df['IdSerie'].isin(priority_list)]
        
        # If matching rows are found, select one at random and return it
        if not filtered_df.empty:
            return filtered_df.sample(n=1, random_state=random.randint(0, 10000)).iloc[0]
    
    # If no rows match any priority, return None
    return None


def generar_prioridades(escritorio_skills):

    unique_skills = set()
    for skills in escritorio_skills.values():
        unique_skills.update(skills)
    
    # Priority levels
    priority_levels = ['alta', 'media', 'baja']
    
    # Generate new dictionary with random priority levels for each unique skill
    skill_priority_dict = {skill: random.choice(priority_levels) for skill in unique_skills}
    
    return skill_priority_dict


def remove_selected_row(df, selected_row):
    """
    Removes the specified row from the DataFrame.
    """
    # If DataFrame or selected_row is None, return original DataFrame
    if df is None or selected_row is None:
        return df
    
    # Get the index of the row to be removed
    row_index = selected_row.name
    
    # Remove the row using the 'drop' method
    updated_df = df.drop(index=row_index)
    
    return updated_df

from typing import Dict, List
from datetime import timedelta


def sort_by_time(nuevos_escritorios_programados: List[Dict[str, any]]) -> List[Dict[str, any]]:

    return sorted(nuevos_escritorios_programados, key=lambda x: x['hora'])



def simular(agenda, niveles_servicio_x_serie, un_dia, prioridades):

    skills                   = extract_first_skills(agenda) #obtener_skills(un_dia)
    configuraciones          = {k:np.random.choice(["Aternancia", "FIFO", "Rebalse"], p=[.5,.25,.25]) for k in skills}
    SLAs                     = [(0.8, 25),(0.9, 30),(.5, 15), (.9, 5),(.5, 25)]
    series = set()
    for sk in skills.values():
        series.update(sk)
    #niveles_servicio_x_serie = {s: random.choice(SLAs) for s in series}
    
    svisor                 = MisEscritorios(skills= skills, configuraciones = configuraciones, niveles_servicio_x_serie = niveles_servicio_x_serie)
    #svisor.actualizar_conexiones(generar_conexiones(svisor))
    tiempo_inicial         = list(generador_emisiones(un_dia))[0].FH_Emi#[0]['FH_Emi']
    generador_emisiones_in = generador_emisiones(un_dia)
    contador_tiempo        = timestamp_iterator(tiempo_inicial)
    reloj_simulacion       = next(contador_tiempo)
    avanzar_un_minuto      = False
    fila                   = pd.DataFrame()
    simulacion             = pd.DataFrame()
    SLA_df                 = pd.DataFrame()
    SLA_index              = 0
    una_emision            = next(generador_emisiones_in)
    emi                    = una_emision['FH_Emi']


    #import copy
    num_emisiones = 2000
    for numero_emision in range(num_emisiones):
        svisor.aplicar_agenda(hora_actual=  reloj_simulacion, agenda = agenda)
        svisor.lanzar_escritorios_programados(reloj_simulacion)
        svisor.actualizar_registos_x_escris(reloj_simulacion)
        print(f"registro hora_actual: {svisor.hora_actual}")
        #si la emision no está en el mismo minuto solo avanzamos los contadores y el reloj:
        if not mismo_minuto(emi, reloj_simulacion):

            print(f"diferencia entre el simulador ({reloj_simulacion}) y la emisión ({emi}) mayor a un minuto: {abs(emi - reloj_simulacion).total_seconds()} s.")
            #avanzar reloj
            reloj_simulacion  = next(contador_tiempo)
            #flag seteada avanzar
            print(f"avanza el reloj un minuto, nuevo tiempo: {reloj_simulacion}, avanza tiempo de espera y tiempo en escritorios en servicio (bloqueados)")
            #print("tiempos de espera incrementados en un minuto")
            fila['espera'] += 1
            if (svisor.filtrar_x_estado('atención') or  svisor.filtrar_x_estado('pausa')):
                en_atencion            = svisor.filtrar_x_estado('atención') or []
                en_pausa               = svisor.filtrar_x_estado('pausa') or []
                escritorios_bloqueados = set(en_atencion + en_pausa)            
                print(f"escritorios ocupados (bloqueados) por servicio: {escritorios_bloqueados}")
                #Avanzar un minuto en todos los tiempos de atención en todos los escritorios bloquedos            
                svisor.iterar_escritorios_bloqueados(escritorios_bloqueados)

            if disponibles:= svisor.filtrar_x_estado('disponible'):
                conectados_disponibles       = [k for k,v in svisor.escritorios_ON.items() if k in disponibles]
                svisor.iterar_escritorios_disponibles(conectados_disponibles)


        #si estamos en el mismo minuto pasar a la siguiente emisión y no correr el reloj
        else:
            print(f"emisioń dentro del mismo minuto (diferencia {abs(emi-reloj_simulacion).total_seconds()} seg.), actualizamos fila, gestionamos escritorios y pasamos a la siguiente emisión")    
            #poner la emisión en una fila de df
            emision_cliente = pd.DataFrame(una_emision).T
            #insertar una nueva columna de tiempo de espera y asignar con cero.
            emision_cliente['espera'] = 0
            #concatenar la emisión a la fila de espera
            fila = pd.concat([fila, emision_cliente]).reset_index(drop=True)
            #fila_para_SLA = copy.deepcopy(fila[['FH_Emi', 'IdSerie', 'espera']])
            print(f"fila actualizada: \n{fila[['FH_Emi','IdSerie','espera']]}")

            if not fila.empty:
                if disponibles:= svisor.filtrar_x_estado('disponible'):
                    #extraer las skills de los escritorios conectados que están disponibles
                    #conectados_disponibles       = [k for k,v in svisor.escritorios_ON.items() if k in disponibles]

                    conectados_disponibles       = balancear_carga_escritorios(
                                                                                {k: {'numero_de_atenciones':v['numero_de_atenciones'],
                                                                                    'tiempo_actual_disponible': v['tiempo_actual_disponible']} 
                                                                                for k,v in svisor.escritorios_ON.items() if k in disponibles}
                                                                                )    

                    skills_disponibles           = {k:v['skills'] for k,v in svisor.escritorios_ON.items() if k in disponibles}
                    configuraciones_disponibles  = {k:v['configuracion_atencion'] for k,v in svisor.escritorios_ON.items() if k in disponibles}
                    print(f"escritorio conectados que están disponibles: {conectados_disponibles}")
                    print(f"skills_disponibles:{skills_disponibles}")
                    print(f"configuraciones_disponibles: {configuraciones_disponibles}")
                    for un_escritorio in conectados_disponibles:

                        configuracion_atencion = svisor.escritorios_ON[un_escritorio]['configuracion_atencion']
                        print(f"buscando cliente para {un_escritorio} con {configuracion_atencion}")
                        fila_filtrada          = fila[fila['IdSerie'].isin(svisor.escritorios_ON[un_escritorio].get('skills', []))]#filtrar_fila_por_skills(fila, svisor.escritorios_ON[un_escritorio])
                        print(f"en base a las skills: {svisor.escritorios_ON[un_escritorio].get('skills', [])}, fila_filtrada \n{fila_filtrada}")
                        if  fila_filtrada.empty:
                                print("No hay match entre idSeries en fila y skills del escritorio, saltar al siguiente escritorio")
                                continue #
                        elif configuracion_atencion == "Aternancia":
                            print("----Aternancia------")
                            print(
                                f"prioridades: {prioridades} skills: {svisor.escritorios_ON[un_escritorio]['skills']} \n{svisor.escritorios_ON[un_escritorio]['pasos_alternancia'].pasos}"                       
                            )                        
                            cliente_seleccionado = svisor.escritorios_ON[un_escritorio]['pasos_alternancia'].buscar_cliente(fila_filtrada)
                            #break
                        elif configuracion_atencion == "FIFO":
                            cliente_seleccionado = FIFO(fila_filtrada)
                            print(f"cliente_seleccionado por {un_escritorio} en configuración FIFO: su emisión fue a las: {cliente_seleccionado.FH_Emi}")
                            #break
                        elif configuracion_atencion == "Rebalse":
                            cliente_seleccionado = extract_highest_priority_and_earliest_time_row(fila_filtrada, prioridades)
                            print(f"cliente_seleccionado por {un_escritorio} en configuración Rebalse: su emisión fue a las: {cliente_seleccionado.FH_Emi}")
                        fila = remove_selected_row(fila, cliente_seleccionado)
                        svisor.iniciar_atencion(un_escritorio, cliente_seleccionado)
                        un_cliente   = pd.DataFrame(cliente_seleccionado[['FH_Emi', 'IdSerie', 'espera']]).T
                        simulacion   =  pd.concat([simulacion, un_cliente])#.reset_index(drop=True)
                        
                        
                        #SLA_una_emision  =  pd.DataFrame(list(reporte_SLA_x_serie(simulacion, 35).items()), columns=['keys', 'values'])
                        
                        SLA_una_emision  =  pd.DataFrame(list(nivel_atencion_x_serie(simulacion, niveles_servicio_x_serie).items()), columns=['keys', 'values'])

                        
                        SLA_index+=1
                        #SLA_index = reloj_simulacion
                        
                        SLA_una_emision['index']            = SLA_una_emision.shape[0]*[SLA_index]
                        SLA_una_emision['hora'] = reloj_simulacion.time().strftime('%H:%M:%S')
                        SLA_df                              = pd.concat([SLA_df, SLA_una_emision], ignore_index=True)#.reset_index(drop=True)
            try:
                #Iterar a la siguiente emisión
                #svisor.aplicar_agenda(hora_actual=  reloj_simulacion, agenda = agenda)    
                una_emision            = next(generador_emisiones_in)
                emi                    = una_emision['FH_Emi']
                print(f"siguiente emisión {emi}")
            except StopIteration:
                print(f"-----------------------------Se acabaron las emisiones en la emision numero {numero_emision} ---------------------------")
                break   

    registros_SLA = SLA_df.pivot(index=['index','hora'], columns=['keys'], values='values').rename_axis(None, axis=1)#.drop_duplicates(subset=['emisión'])

    df_reset = registros_SLA.reset_index()
    duplicates =  df_reset['hora'].duplicated(keep=False)
    registros_SLA = df_reset[~duplicates].drop('index', axis=1,inplace=False).reset_index(drop=True, inplace=False)

    return registros_SLA

def timestamp_iterator(initial_timestamp: str):
    current_time = initial_timestamp
    while True:
        yield current_time#.strftime('%Y-%m-%d %H:%M:%S')        
        current_time += timedelta(minutes=1)

def generar_conexiones(svisor, p_true:float=4, p_false:float=1):
    return {f"{key}": random.choices([True, False], [p_true, p_false])[0] for key in {**svisor.escritorios_ON, **svisor.escritorios_OFF}}

from datetime import datetime, timedelta
from random import randint
def mismo_minuto(tiempo_1, tiempo_2):
    return abs(tiempo_1 - tiempo_2).total_seconds() <= 60


def timestamp_iterator(initial_timestamp: str):
    current_time = initial_timestamp
    while True:
        yield current_time#.strftime('%Y-%m-%d %H:%M:%S')        
        current_time += timedelta(minutes=1)
        
def generador_emisiones(df):
    #for _ , self.row in self.df.iterrows():
    for _ , df.row in df.iterrows():
        yield df.row  # Yield each row as a pandas Series
        
        
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def graficar(df):
    
    fixed_date = "2000-01-01"
    df['hora'] = pd.to_datetime(df['hora'].astype(str).apply(lambda x: f"{fixed_date} {x}"))
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(12, 3))
    # Plot each column against 'hora'
    for column in df.columns:
        if column != 'hora':
            ax.plot(df['hora'], df[column], label=column)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    # Configure y-axis to show only the hour
    y_labels = ax.get_yticks()
    #y_labels = [f"{int(label)}:00" for label in y_labels]
    ax.set_yticklabels(y_labels)
    # Add labels and title
    plt.xlabel('Hora')
    plt.ylabel('%')
    plt.title('Niveles de atención')
    plt.legend()
    plt.show()