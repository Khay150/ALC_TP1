import numpy as np
from scipy.linalg import solve_triangular


def calcularLU(A):
    dim = A.shape[0]
    I = np.eye(dim).astype(float)
    tau = np.zeros([dim, 1]).astype(float)
    ek = np.zeros([1, dim]).astype(float)
    mk = np.zeros([dim, dim]).astype(float)
    U = A.astype(float)
    L = np.eye(dim)
    P = np.eye(dim)
    
    
    for fila in range(dim-1):

        max_index = np.argmax(np.abs(U[fila:, fila])) + fila
        if max_index != fila:
            U[[fila, max_index], :] = U[[max_index, fila], :]
            P[[fila, max_index], :] = P[[max_index, fila], :]
            
            L[[fila, max_index], :fila] = L[[max_index, fila], :fila]

        for rep in range(dim-fila-1):
            tau[fila+rep+1][0] = U[fila+rep+1][fila] / U[fila][fila]

        ek[0][fila] = 1
        tauxek = tau @ ek
        mk = I - tauxek
        U = mk @ U
        mkInv = I + tauxek
        L = L @ mkInv
        ek = np.zeros([1, dim]).astype(float)
        tau = np.zeros([dim, 1]).astype(float)
    
    return P, L, U



def inversaLU(matriz):
    
    P, L, U = calcularLU(matriz)
    
    dim = L.shape[0]

    UINV = np.zeros([dim, dim]).astype(float)
    LINV = np.zeros([dim, dim]).astype(float)
    
    I_col = np.zeros([dim,1])
    
    for columna in range(dim):
       
       I_col[columna][0] = 1

       UINV_col = solve_triangular(U, I_col, lower=False)
       LINV_col = solve_triangular(L, I_col, lower=True)
       
       
       for j in range(dim):
            UINV[j][columna] = UINV_col[j][0]
            LINV[j][columna] = LINV_col[j][0]
       
       I_col[columna][0] = 0
   
    return UINV@LINV@P


