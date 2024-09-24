#%% import de funciones
import pandas as pd
from inline_sql import sql, sql_val
import numpy as np
from funciones import inversaLU
import matplotlib.pyplot as plt

#%% Leemos el excel



datos_basicos = pd.read_excel("matrizlatina2011_compressed_0.xlsx", sheet_name= "LAC_IOT_2011")


#Separamos los datos de las filas de nuestros 2 paises
datos_impotantes_rel= sql^ """
                        SELECT *
                        FROM datos_basicos
                        WHERE Country_iso3 LIKE 'CRI%' or Country_iso3 LIKE 'VEN%' 
                    """ 
                    
#%% Contruimos una tabla auxiliar con todos los datos
#Creamos un dataframe con los paises y los sectores
datos_relacion_iso = pd.concat([datos_impotantes_rel['Country_iso3'], datos_impotantes_rel['Nosector']], axis = 1)              
#filtramos los datos de las columnas de relacion con Venezuela
datos_relacion_filtrados = pd.concat([datos_relacion_iso, datos_impotantes_rel.filter(regex="^VEN").copy()], axis = 1)
#filtramos los datos de las columnas de relacion con Costa Rica
datos_relacion= pd.concat([datos_relacion_filtrados, datos_impotantes_rel.filter(regex="^CRI").copy()], axis = 1)
#agregamos los datos del output
datos_relacion = pd.concat([datos_relacion, datos_impotantes_rel['Output']], axis = 1)

#%% Solucion a los output=0
#las columnas que involucran a ambos paises
columnas = pd.concat([datos_relacion.filter(regex="^CRI"), datos_relacion.filter(regex="^VEN")], axis = 1).columns
# Procesamos los datos un poco porque nos encontramos con la dificultad de que los output valian 0 en esas columnas.
# Pero los valores en dichas filas no eran exclusivamente 0. Entonces los igualamos a 0, ya que eran del orden de 10^(-5)

datos_relacion.loc[(datos_relacion['Country_iso3'] == 'VEN') & (datos_relacion['Nosector'] == 's28'), columnas] = 0

datos_relacion.loc[(datos_relacion['Country_iso3'] == 'VEN') & (datos_relacion['Nosector'] == 's31'), columnas] = 0

datos_relacion.loc[(datos_relacion['Country_iso3'] == 'CRI') & (datos_relacion['Nosector'] == 's31'), columnas] = 0

datos_relacion.loc[(datos_relacion['Country_iso3'] == 'CRI') & (datos_relacion['Nosector'] == 's03'), columnas] = 0

#%% Separamos los datos en 4 dataframes
# Construimos un df con Todos los valores de todos los sectores de ambos paises
numeros_relacion = pd.concat([datos_relacion.filter(regex="^CRI"),  datos_relacion.filter(regex="^VEN")], axis = 1)
#Sepaaramos las filas solo de Costa Rica
numeros_CRI = sql^ """
                        SELECT *
                        FROM datos_relacion
                        WHERE Country_iso3 LIKE 'CRI%'
                    """ 
#Datos de Costa Rica vs Costa Rica
numeros_internos_CRI = numeros_CRI.filter(regex="^CRI")
#Datos de Costa Rica vs Venezuela
numeros_externos_CRI = numeros_CRI.filter(regex="^VEN")
#Analogamente con los de Venezuela
numeros_VEN = sql^ """
                        SELECT *
                        FROM datos_relacion
                        WHERE Country_iso3 LIKE 'VEN%'
                    """ 

numeros_internos_VEN = numeros_VEN.filter(regex="^VEN")

numeros_externos_VEN = numeros_VEN.filter(regex="^CRI")

#%% Pasamos de DataFrame a np.array



Z_CRICRI = numeros_internos_CRI.to_numpy()

Z_VENVEN = numeros_internos_VEN.to_numpy()


Z_CRIVEN = numeros_externos_CRI.to_numpy()

Z_VENCRI = numeros_externos_VEN.to_numpy()


matirz_Z_grande = numeros_relacion.to_numpy()


#%% Conseguimos las Matrices P
P_VEN=np.zeros([40,40])
#Tomamos las columnas Output
Output_VEN = sql^ """
                        SELECT Output
                        FROM datos_relacion
                        WHERE Country_iso3 LIKE 'VEN%'
                    """
                    
Output_VEN = Output_VEN.to_numpy()
#Construimos P para Venezuela
for fila in range (P_VEN.shape[0]):
    if Output_VEN[fila]==0:
        P_VEN[fila][fila]=1
    else:
        P_VEN[fila][fila]= Output_VEN[fila]

#De la misma manera para Costa Rica
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

#%% Resolución ejercicio 7

#Tomamos las inversas
P_INV_CRI=inversaLU(P_CRI)
P_INV_VEN=inversaLU(P_VEN)

# Ejercicio 7 Matrices Insumo-Producto
A_VV= Z_VENVEN@P_INV_VEN

A_VC = Z_VENCRI@P_INV_CRI

A_CC= Z_CRICRI@P_INV_CRI
A_CV = Z_CRIVEN@P_INV_VEN

#%% Completamos la A del sistema completo

A_grande = np.zeros([80,80])

for fila in range(80):
    for columna in range(80):
        if (fila < 40) and (columna < 40):
            A_grande[fila][columna] = A_VV[fila][columna]
        elif (fila < 40):
            A_grande[fila][columna] = A_VC[fila][columna -40]
        elif (fila >= 40) and (columna < 40):
            A_grande[fila][columna] = A_CV[fila -40][columna]
        else:
            A_grande[fila][columna] = A_CC[fila -40][columna -40]

#%% Simulamos el shock

I = np.eye(40)
#Calculamos la demanda
d_VEN_simple = (I- A_VV)@Output_VEN
#Calulcamos el shock en la demanda
d_VEN_shock_simple = d_VEN_simple.copy()

d_VEN_shock_simple[4] = d_VEN_shock_simple[4]*0.9

d_VEN_shock_simple[5] = d_VEN_shock_simple[5]*1.033
d_VEN_shock_simple[6] = d_VEN_shock_simple[6]*1.033
d_VEN_shock_simple[7] = d_VEN_shock_simple[7]*1.033


#Calculo del delta
delta_d_simple = d_VEN_shock_simple - d_VEN_simple
#Calculamos la producción segun lo desarollado en consigna  4
delta_p_simple= inversaLU(I - A_VV) @ delta_d_simple

p_prima_simple = Output_VEN + delta_p_simple

#%% Simulamos el shock para el modelo completo

I = np.eye(40)
#Calculamos la demanda de venezuela
d_VEN_completo = (I- A_VV)@Output_VEN
#Calculamos el shock
d_VEN_shock_completo = d_VEN_completo.copy()

d_VEN_shock_completo[4] = d_VEN_shock_completo[4]*0.9

d_VEN_shock_completo[5] = d_VEN_shock_completo[5]*1.033
d_VEN_shock_completo[6] = d_VEN_shock_completo[6]*1.033
d_VEN_shock_completo[7] = d_VEN_shock_completo[7]*1.033

#sacamos el delta
delta_d_completo = d_VEN_shock_completo - d_VEN_completo
#usamos la consigna 6 ya que la demanda Costa Rica no se modifica
delta_p_completo = inversaLU(I-A_VV - A_VC @ inversaLU(I - A_CC) @ A_CV) @ delta_d_completo

p_prima_completo = Output_VEN + delta_p_completo


#%% Realizamos un grafico de barras comparando los resultados para el modelo simple
x = np.arange(1, 41)

# Definir el ancho de las barras
width = 0.35

# Crear el gráfico de barras
fig, ax = plt.subplots()

# Dibujar las barras del primer array
bars1 = ax.bar(x - width/2, np.ravel(Output_VEN), width, label='Output_VEN')

# Dibujar las barras del segundo array
bars2 = ax.bar(x + width/2, np.ravel(p_prima_simple), width, label='p_prima_simple')

ax.set_xlabel('Fila')
ax.set_ylabel('Valores')
ax.set_title('Comparación de Arrays')
ax.set_xticks(x)
ax.set_xticklabels([f'{i}' for i in x])

# Añadir leyenda
ax.legend()

# Mostrar el gráfico
plt.show()

#%% Realizamos un grafico de barras comparando los resultados para el modelo completo

x = np.arange(1, 41)

# Definir el ancho de las barras
width = 0.35

# Crear el gráfico de barras
fig, ax = plt.subplots()

# Dibujar las barras del primer array
bars1 = ax.bar(x - width/2, np.ravel(Output_VEN), width, label='Output_VEN')

# Dibujar las barras del segundo array
bars2 = ax.bar(x + width/2, np.ravel(p_prima_completo), width, label='p_prima_completo')

ax.set_xlabel('Fila')
ax.set_ylabel('Valores')
ax.set_title('Comparación de Arrays')
ax.set_xticks(x)
ax.set_xticklabels([f'{i}' for i in x])

# Añadir leyenda
ax.legend()

# Mostrar el gráfico
plt.show()

#%% Y comparando ambas variaciones entre si 

x = np.arange(1, 41)

# Definir el ancho de las barras
width = 0.35

# Crear el gráfico de barras
fig, ax = plt.subplots()

# Dibujar las barras del primer array
bars1 = ax.bar(x - width/2, np.ravel(p_prima_simple), width, label='p_prima_simple')

# Dibujar las barras del segundo array
bars2 = ax.bar(x + width/2, np.ravel(p_prima_completo), width, label='p_prima_completo')

ax.set_xlabel('Fila')
ax.set_ylabel('Valores')
ax.set_title('Comparación de Arrays')
ax.set_xticks(x)
ax.set_xticklabels([f'{i}' for i in x])

# Añadir leyenda
ax.legend()

# Mostrar el gráfico
plt.show()


