import numpy as np
from scipy.linalg import solve_triangular


def calcularLU(A):
    # Inicializa variables
    dim = A.shape[0] # Dimensión de la matriz A
    I = np.eye(dim).astype(float) # Matriz identidad de tamaño 'dim' 
    tau = np.zeros([dim, 1]).astype(float) # Vector tau para almacenar factores de escala
    ek = np.zeros([1, dim]).astype(float) # Vector fila auxiliar
    mk = np.zeros([dim, dim]).astype(float) # Matriz auxiliar para actualizar U y L
    U = A.copy().astype(float) # Matriz U, inicialmente es igual a A
    L = np.eye(dim) # Inicialmente, L es la matriz identidad
    P = np.eye(dim) # Matriz de permutación inicializada como identidad
    
    # Ciclo principal para la factorización
    for fila in range(dim-1):
        # Pivoteo parcial: se busca el valor absoluto máximo en la columna 'fila' de U para intercambiar filas
        max_index = np.argmax(np.abs(U[fila:, fila])) + fila # Encuentra la fila con el valor máximo
        if max_index != fila:
            # Si la fila máxima no es la fila actual, se intercambian las filas en U, P y L
            U[[fila, max_index], :] = U[[max_index, fila], :]
            P[[fila, max_index], :] = P[[max_index, fila], :]
            L[[fila, max_index], :fila] = L[[max_index, fila], :fila]

        # Calcular los coeficientes multiplicadores para las filas debajo de la fila actual
        for rep in range(dim-fila-1):
            # tau almacena los factores de eliminación que se usarán para hacer ceros en U
            tau[fila+rep+1][0] = U[fila+rep+1][fila] / U[fila][fila]

        # Construcción de una matriz de eliminación con ek y tau para hacer ceros en la columna actual de U
        ek[0][fila] = 1 # Vector columna con un 1 en la posición de la fila actual
        tauxek = tau @ ek # Producto de tau y ek que se usará para actualizar mk
        mk = I - tauxek # Matriz que realiza la eliminación en U
        U = mk @ U # Aplica la eliminación a U
        
        # Actualización de la matriz L
        mkInv = I + tauxek # Matriz inversa de mk para actualizar L
        L = L @ mkInv # Aplica la actualización de L
        
        # Reinicia los vectores ek y tau para la siguiente iteración
        ek = np.zeros([1, dim]).astype(float)
        tau = np.zeros([dim, 1]).astype(float)
    
    # Devuelve las matrices de permutación P, triangular inferior L y triangular superior U
    return P, L, U



def inversaLU(matriz):
    
    P, L, U = calcularLU(matriz) # Calcula la descomposición LU de la matriz de entrada
    
    dim = L.shape[0] # Dimensión de la matriz

    # Inicializa matrices para almacenar la inversa de U y L
    UINV = np.zeros([dim, dim]).astype(float) # Matriz inversa de U
    LINV = np.zeros([dim, dim]).astype(float) # Matriz inversa de L
    
    I_col = np.zeros([dim,1]) # Vector columna que representará las columnas de la identidad
    
    # Ciclo para calcular las inversas de L y U columna por columna
    for columna in range(dim):
       
       I_col[columna][0] = 1 # Establece la columna correspondiente de la identidad

       # Resuelve el sistema triangular superior para U usando la columna de la identidad
       UINV_col = solve_triangular(U, I_col, lower=False) # Soluciona Ux = I_col para encontrar una columna de la inversa de U
       
       # Resuelve el sistema triangular inferior para L usando la columna de la identidad
       LINV_col = solve_triangular(L, I_col, lower=True) # Soluciona Lx = I_col para encontrar una columna de la inversa de L
       
       # Asigna los resultados obtenidos a las matrices inversas UINV y LINV
       for j in range(dim):
            UINV[j][columna] = UINV_col[j][0] # Asigna los valores de la columna invertida de U
            LINV[j][columna] = LINV_col[j][0] # Asigna los valores de la columna invertida de L
       
       I_col[columna][0] = 0 # Reinicia la columna de la identidad a ceros para la siguiente iteración
   
    return UINV@LINV@P # Devuelve la matriz inversa de la matriz original, que es el producto de UINV, LINV y P


