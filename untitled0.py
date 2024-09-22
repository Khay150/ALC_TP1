

import pandas as pd
from inline_sql import sql, sql_val
import numpy as np
from funciones import inversaLU

#%%
carpeta = "D:/FRAN/Educacion/UBA/ALC/ALC_TP1/"

datos_basicos = pd.read_excel("matrizlatina2011_compressed_0.xlsx", sheet_name= "LAC_IOT_2011")



datos_impotantes_rel= sql^ """
                        SELECT *
                        FROM datos_basicos
                        WHERE Country_iso3 LIKE 'CRI%' or Country_iso3 LIKE 'VEN%' 
                    """ 
                    
#%%
datos_relacion_iso = pd.concat([datos_impotantes_rel['Country_iso3'], datos_impotantes_rel['Nosector']], axis = 1)              

datos_relacion_filtrados = pd.concat([datos_relacion_iso, datos_impotantes_rel.filter(regex="^VEN").copy()], axis = 1)

datos_relacion= pd.concat([datos_relacion_filtrados, datos_impotantes_rel.filter(regex="^CRI").copy()], axis = 1)

datos_relacion = pd.concat([datos_relacion, datos_impotantes_rel['Output']], axis = 1)

#%%

columnas = pd.concat([datos_relacion.filter(regex="^CRI"), datos_relacion.filter(regex="^VEN")], axis = 1).columns

datos_relacion.loc[(datos_relacion['Country_iso3'] == 'VEN') & (datos_relacion['Nosector'] == 's28'), columnas] = 0

datos_relacion.loc[(datos_relacion['Country_iso3'] == 'VEN') & (datos_relacion['Nosector'] == 's31'), columnas] = 0

datos_relacion.loc[(datos_relacion['Country_iso3'] == 'CRI') & (datos_relacion['Nosector'] == 's31'), columnas] = 0

datos_relacion.loc[(datos_relacion['Country_iso3'] == 'CRI') & (datos_relacion['Nosector'] == 's03'), columnas] = 0

#%%

numeros_relacion = pd.concat([datos_relacion.filter(regex="^CRI"),  datos_relacion.filter(regex="^VEN")], axis = 1)

numeros_CRI = sql^ """
                        SELECT *
                        FROM datos_relacion
                        WHERE Country_iso3 LIKE 'CRI%'
                    """ 

numeros_internos_CRI = numeros_CRI.filter(regex="^CRI")

numeros_VEN = sql^ """
                        SELECT *
                        FROM datos_relacion
                        WHERE Country_iso3 LIKE 'VEN%'
                    """ 

numeros_internos_VEN = numeros_VEN.filter(regex="^VEN")

#%%

matriz_CRI = numeros_internos_CRI.to_numpy()

matriz_VEN = numeros_internos_VEN.to_numpy()

matriz_relacion = numeros_relacion.to_numpy()


#%%
P_VEN=np.zeros([40,40])

Output_VEN = sql^ """
                        SELECT Output
                        FROM datos_relacion
                        WHERE Country_iso3 LIKE 'VEN%'
                    """
                    
Output_VEN = Output_VEN.to_numpy()
for fila in range (P_VEN.shape[0]):
    if Output_VEN[fila]==0:
        P_VEN[fila][fila]=1
    else:
        P_VEN[fila][fila]=Output_VEN[fila]


P_CRI=np.zeros([40,40])

Output_CRI= sql^ """
                        SELECT Output
                        FROM datos_relacion
                        WHERE Country_iso3 LIKE 'CRI%'
                    """
                    
Output_CRI = Output_CRI.to_numpy()
for fila in range (P_CRI.shape[0]):
    if Output_CRI[fila]==0:
        P_CRI[fila][fila]=1
    else:
        P_CRI[fila][fila]=Output_CRI[fila]

#%%


P_INV_CRI=inversaLU(P_CRI)
P_INV_VEN=inversaLU(P_VEN)

# Ejercicio 7 Matrices Insumo-Producto
A_VV=matriz_VEN@P_INV_VEN
A_CC=matriz_CRI@P_INV_CRI

#%%