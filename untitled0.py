

import pandas as pd
from inline_sql import sql, sql_val
import numpy as np


carpeta = "D:/FRAN/Educacion/UBA/ALC/ALC_TP1/"

datos_basicos = pd.read_excel("matrizlatina2011_compressed_0.xlsx", sheet_name= "LAC_IOT_2011")

datos_impotantes_CRI = sql^ """
                        SELECT *
                        FROM datos_basicos
                        WHERE Country_iso3 LIKE 'CRI%' 
                    """ 
                    
datos_impotantes_VEN = sql^ """
                        SELECT *
                        FROM datos_basicos
                        WHERE Country_iso3 LIKE 'VEN%' 
                    """ 
                    
datos_internos_CRI = datos_impotantes_CRI.filter(regex="^CRI").copy()

datos_finales_CRI = pd.concat([datos_impotantes_CRI['Nosector'], datos_internos_CRI], axis = 1)

datos_internos_VEN = datos_impotantes_CRI.filter(regex="^CRI").copy()

datos_finales_VEN = pd.concat([datos_impotantes_VEN['Nosector'], datos_internos_VEN], axis = 1)


datos_otuput_VEN = sql^ """
                        SELECT Country_iso3, Nosector, Output
                        FROM datos_basicos
                        WHERE Country_iso3 LIKE 'VEN%' 
                    """ 
                    
datos_otuput_CRI = sql^ """
                        SELECT Country_iso3, Nosector, Output
                        FROM datos_basicos
                        WHERE Country_iso3 LIKE 'CRI%' 
                    """ 
                    
datos_VEN = sql^ """
                        SELECT *
                        FROM datos_basicos
                        WHERE Country_iso3 LIKE 'VEN%' AND Output = 0
                    """ 