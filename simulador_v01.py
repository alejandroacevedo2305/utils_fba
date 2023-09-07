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
import warnings
warnings.filterwarnings("ignore")

class datos():
    def __init__(self, db: str= 'simulation_db.sqlite'):
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

    def un_mes(self, mes:str=3, col_emisiones:str='FH_Emi'):

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

#DT = datos(db='IA_simulador/mock_db.sqlite').un_mes().un_dia()
#DT.df # TIENE QUE EXISTIR --- POR MOTIVOS
# DT.emisiones_forecast(
#     dia = '2023-06-15',
#     hora_ini = '07:30:00',
#     hora_fin = '16:30:00',
# ) #
#DT.df.info()
#  Entrena un modelo
# DT.df_forecast # Deberia haber algo aqui
# GENERADOR = DT.emisiones_generador()
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
class MisEscritorios:

    def __init__(self, skills:Dict[str, List[int]], configuraciones: Dict[str, str], conexiones: Dict[str, bool] = None):
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
        self.skills_configuraciones = unir_values_en_tupla(self.skills, self.configuraciones)
        #iteramos self.skills_configuraciones para armar el dicionario con los escritorios
        self.escritorios            = {key: #escritorio i
                                    {
                                    "skills":series, #series cargadas en el escritorio i
                                    'contador_bloqueo':None, #campo vacío donde se asignará un iterador que cuenta los minutos que el escritorio estará ocupado en servicio
                                    'minutos_bloqueo':None, #campo vacío donde se asignarán los minutos que se demora la atención
                                    'estado':'disponible', #si el escritorio está o no disponible para atender
                                    'configuracion_atencion':config, #configuración del tipo de antención, FIFO, RR, etc.
                                    'conexion':None, #campo vacío donde se asignará el estado de la conexión del escritorio (ON/OFF)
                                    } for key,(series, config) in self.skills_configuraciones.items()}

        if not conexiones:
            #si no se provee el estado de los conexiones se asumen todas como True (todos conectados):
            conexiones = {f"{key}": random.choices([True, False], [1, 0])[0] for key in self.escritorios}
        #Al inicializar el objeto asignamos las conexiones
        self.escritorios                          = actualizar_conexiones(self.escritorios, conexiones)
        #Luego separamos los escritorios en conectados (self.escritorios_ON) y desconectados (self.escritorios_OFF)
        self.escritorios_ON, self.escritorios_OFF = separar_por_conexion(self.escritorios)
        self.nuevos_escritorios_programados       = []

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


    def cambiar_propiedades_escritorio(self, escritorio:str, skills:List[int]=None, configuracion_atencion:str=None, conexion:bool=None) -> None:
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

    def iniciar_bloqueo(self, escritorios_a_bloqueo:  List[str], emision:list):
        """_summary_

        Args:
            escritorios_a_bloqueo (List[str]): lista de escritorios que serán bloqueados por atención.
            emision (list): emision del cliente que será atendido
        """

        # iterar los escritorios y emisiones
        for escr_bloq, emi in zip(escritorios_a_bloqueo, emision):

            #extraer los minutos que dura la atención asociada a la emision
            n_minutos = round((emi.FH_AteFin - emi.FH_AteIni).total_seconds()/60)
            #reescribir campos:

            self.escritorios_ON[escr_bloq]['contador_bloqueo'] = iter(islice(count(start=0, step=1), n_minutos))#nuevo contador de minutos limitado por n_minutos
            self.escritorios_ON[escr_bloq]['estado']           = 'bloqueado'#estado bloqueado significa que está atendiendo al cliente.
            self.escritorios_ON[escr_bloq]['minutos_bloqueo']  = n_minutos#tiempo de atención

    def filtrar_x_disponibilidad(self, state: str):
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
            return None

    def iterar_escritorio_bloqueado(self, escritorios_bloqueados: List[str]):
        """_summary_

        Args:
            escritorios_bloqueados (List[str]): lista de escritorios bloqueados para iterar los minutos.

        Returns:
            _type_: _description_
        """
        if not self.filtrar_x_disponibilidad('bloqueado'):
            print("no hay escritorios bloqueados")
            return None
        else:

            for escri_bloq in escritorios_bloqueados:
                #extraer contador
                value = next(self.escritorios_ON[escri_bloq]['contador_bloqueo'], None)
                #si está vació es por que terminó
                if value is None:
                    self.escritorios_ON[escri_bloq]['estado'] = 'disponible'
                    print(f"escritorio {escri_bloq} disponible")
                #Si no está vació es por que tiene que seguir iterando.
                #Abajo se imprime la diferencia entre los minutos totales de atención y el valor de la iteración,
                #es decir, el tiempo que le queda en servicio.
                else:
                    print(f"Escritorio {escri_bloq}: quedan {(self.escritorios_ON[escri_bloq]['minutos_bloqueo'])-(value)} min. de atención")

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

        #return self


def timestamp_iterator(initial_timestamp: str):
    current_time = initial_timestamp
    while True:
        yield current_time#.strftime('%Y-%m-%d %H:%M:%S')
        current_time += timedelta(minutes=1)

# un_dia                 = datos().un_mes().un_dia().df#[un_dia.IdOficina == 10]
# un_dia.sort_values(by='FH_Emi', inplace=True)
# un_dia                 = un_dia[un_dia.IdOficina == 10]
# skills                 = obtener_skills(un_dia)
# prioridades            = generar_prioridades(skills)
# configuraciones        = {k:np.random.choice(["x_t_de_Espera", "FIFO", "x_Prioridad_de_serie"], p=[.4,.4,.2]) for k in skills}
# svisor                 = MisEscritorios(skills, configuraciones)
# def generador_emisiones(df):
#     for _ , df.row in df.iterrows():
#         yield df.row  # Yield each row as a pandas Series

# tiempo_inicial         = list(generador_emisiones(un_dia))[0]['FH_Emi']

#from datetime import timedelta

# contador_tiempo        = timestamp_iterator(tiempo_inicial)

# for _ in range(900):

#     reloj_simulacion       = next(contador_tiempo)
#     svisor.lanzar_escritorios_programados(reloj_simulacion)




    # for propiedades in nuevos_escritorios_programados:


    #     config_time = pd.Timestamp(f"{reloj_simulacion.date()} {propiedades['hora']}")
    #     #print(config_time)
    #     time_diff = abs(config_time - reloj_simulacion)
    #     #print(time_diff)
    #     if time_diff <= timedelta(minutes=.5):
    #         print(reloj_simulacion, config_time)

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




def simular(generador_emisiones, tiempo_inicial, svisor, prioridades, num_emisiones:int, sla:float):  # sourcery skip: low-code-quality


    contador_tiempo        = timestamp_iterator(tiempo_inicial)
    reloj_simulacion       = next(contador_tiempo)
    una_emision            = next(generador_emisiones)

    emi                    = una_emision['FH_Emi']

    avanzar_un_minuto      = False
    fila                   = pd.DataFrame()
    simulacion             = pd.DataFrame()
    SLA_df                 = pd.DataFrame()
    SLA_index              = 0
    #en tiempo cero el reloj, el tiempo inicial y la primera emisión deben ser las mismas
    assert reloj_simulacion == tiempo_inicial == emi
    print(f'Todo sincronizado: \ntiempo inicial: {tiempo_inicial}, \nreloj simulacióń: {reloj_simulacion}, \nprimera emision: {emi}')
    # if avanzar_un_minuto:
    #     print("tiempos de espera incrementados en un minuto")
    #     fila['espera'] += 1
    #     if escritorios_bloqueados := svisor.filtrar_x_disponibilidad('bloqueado'):
    #         print(f"escritorios ocupados (bloqueados) en servicio: {escritorios_bloqueados}")
    #         #Avanzar un minuto en todos los tiempos de atención en todos los escritorios bloquedos
    #         svisor.iterar_escritorio_bloqueado(escritorios_bloqueados)
    #si la emision no está dentro de un minuto de rango con el reloj:
    for _ in range(num_emisiones):

        svisor.lanzar_escritorios_programados(reloj_simulacion)


        if not mismo_minuto(emi, reloj_simulacion):

            print(f"diferencia entre el simulador ({reloj_simulacion}) y la emisión ({emi}) mayor a un minuto: {abs(emi - reloj_simulacion).total_seconds()} s.")
            #avanzar reloj
            reloj_simulacion  = next(contador_tiempo)
            #flag seteada avanzar
            avanzar_un_minuto = True
            print(f"avanza el reloj un minuto, nuevo tiempo: {reloj_simulacion}, avanza tiempo de espera y tiempo en escritorios en servicio (bloqueados)")

            #print("tiempos de espera incrementados en un minuto")
            fila['espera'] += 1
            if escritorios_bloqueados := svisor.filtrar_x_disponibilidad('bloqueado'):
                print(f"escritorios ocupados (bloqueados) en servicio: {escritorios_bloqueados}")
                #Avanzar un minuto en todos los tiempos de atención en todos los escritorios bloquedos

                svisor.iterar_escritorio_bloqueado(escritorios_bloqueados)

        #si estamos en el mismo minuto pasar a la siguiente emisión y no correr el reloj
        else:
            print(f"emisioń dentro del mismo minuto (diferencia {abs(emi-reloj_simulacion).total_seconds()} seg.), actualizamos fila, gestionamos escritorios y pasamos a la siguiente emisión")
            #poner la emisión en una fila de df
            emision_cliente = pd.DataFrame(una_emision).T
            #insertar una nueva columna de tiempo de espera y asignar con cero.
            emision_cliente['espera'] = 0
            #concatenar la emisión a la fila de espera
            fila = pd.concat([fila, emision_cliente]).reset_index(drop=True)

            print(f"fila actualizada: \n{fila[['FH_Emi','IdSerie','espera']]}")

            if not fila.empty:
                if disponibles:= svisor.filtrar_x_disponibilidad('disponible'):
                    #extraer las skills de los escritorios conectados que están disponibles
                    conectados_disponibles       = [k for k,v in svisor.escritorios_ON.items() if k in disponibles]
                    skills_disponibles           = {k:v['skills'] for k,v in svisor.escritorios_ON.items() if k in disponibles}
                    configuraciones_disponibles  = {k:v['configuracion_atencion'] for k,v in svisor.escritorios_ON.items() if k in disponibles}
                    print(f"escritorio conectados que están disponibles: {conectados_disponibles}")
                    print(f"skills_disponibles:{skills_disponibles}")
                    print(f"configuraciones_disponibles: {configuraciones_disponibles}")
                    for un_escritorio in conectados_disponibles:
                        print(f"asignando cliente para {un_escritorio}")
                        configuracion_atencion = svisor.escritorios_ON[un_escritorio]['configuracion_atencion']
                        fila_filtrada          = filtrar_fila_por_skills(fila, svisor.escritorios_ON[un_escritorio])

                        if  fila_filtrada.empty:
                            print("No hay match entre idSeries en fila y skills del escritorio, saltar al siguiente escritorio")
                            continue #

                        if configuracion_atencion   == 'FIFO':
                            cliente_seleccionado = FIFO(fila_filtrada)
                            print(f"cliente_seleccionado por {un_escritorio} en configuración FIFO: su emisión fue a las: {cliente_seleccionado.FH_Emi}")

                        elif configuracion_atencion == 'x_t_de_Espera':
                            cliente_seleccionado = x_t_de_Espera(fila_filtrada)
                            print(f"cliente_seleccionado por {un_escritorio} en configuración x_t_de_Espera: su emisión fue a las: {cliente_seleccionado.FH_Emi}")

                        elif configuracion_atencion == 'x_Prioridad_de_serie':
                            cliente_seleccionado = x_Prioridad_de_serie(fila_filtrada, prioridades)
                            print(f"cliente_seleccionado por {un_escritorio} en configuración x_Prioridad_de_serie: su emisión fue a las: {cliente_seleccionado.FH_Emi}")

                        svisor.iniciar_bloqueo(escritorios_a_bloqueo = [un_escritorio], emision=[cliente_seleccionado])
                        #escritorios_bloqueados = svisor.filtrar_x_disponibilidad('bloqueado')


                        #antes de eliminar al cliente que más ha esperado lo guardamos
                        registro_simulaciones               = pd.DataFrame(cliente_seleccionado[['FH_Emi', 'IdSerie', 'espera']]).T
                        registro_simulaciones['escritorio'] = un_escritorio

                        registro_simulaciones['atencion']   = svisor.escritorios_ON[un_escritorio]['minutos_bloqueo']
                        simulacion                          =  pd.concat([simulacion, registro_simulaciones]).reset_index(drop=True)

                        #SLA_df                             = pd.concat([SLA_df, pd.DataFrame([reporte_SLA(simulacion, 45)])])#.reset_index(drop=True)
                        SLA_una_emision                     = pd.DataFrame(list(reporte_SLA(simulacion, sla).items()), columns=['keys', 'values'])
                        SLA_index+=1
                        SLA_una_emision['index']            = SLA_una_emision.shape[0]*[SLA_index]
                        SLA_df                              = pd.concat([SLA_df, SLA_una_emision], ignore_index=True)#.reset_index(drop=True)
                        fila = remove_selected_row(fila, cliente_seleccionado)


            try:
                #Iterar a la siguiente emisión
                una_emision            = next(generador_emisiones)
                emi                    = una_emision['FH_Emi']
                #flag seteada para NO avanzar
                avanzar_un_minuto = False
                print(f"siguiente emisión {emi}")

            except StopIteration:
                print("-----------------------------Se acabaron las emisiones.---------------------------")
                break





    return simulacion, SLA_df.pivot(index='index', columns='keys', values='values').rename_axis(None, axis=1)

def generador_emisiones(df):
    #for _ , self.row in self.df.iterrows():
    for _ , df.row in df.iterrows():
        yield df.row  # Yield each row as a pandas Series
