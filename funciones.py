import numpy as np
from scipy.linalg import solve_triangular

A=np.array([[1,4,7],
           [2,5,8],
           [3,6,10]])

def calcularLU(A):
    dim=A.shape[0]
    I=np.eye(dim).astype(float)
    tau=np.zeros([dim,1]).astype(float)
    ek=np.zeros([1,dim]).astype(float)
    mk=np.zeros([dim,dim]).astype(float)
    U=A.astype(float)
    L=np.eye(dim)
    for fila in range(dim-1):
        for rep in range(dim-fila-1):
            tau[fila+rep+1][0]=U[fila+rep+1][fila]/U[fila][fila] #primer elemento de cada fila/pivote

        ek[0][fila]=1
        tauxek=(tau@ek)
        mk=I-tauxek
        U=mk@U
        mkInv=I+tauxek
        L=L@mkInv
        ek=np.zeros([1,dim]).astype(float)
        tau=np.zeros([dim,1]).astype(float)
    return L,U

B= calcularLU(A)

def inversaLU(L, U):
    dim=L.shape[0]

    I_col = np.zeros([dim,1]).astype(float)
    
    UINV, LINV = np.zeros([dim, dim]).astype(float), np.eye(dim).astype(float)
    
    for columna in range(dim):
        
        I_col[columna][0] = 1
        
        UINV_col = solve_triangular(U, I_col)
        LINV_col = solve_triangular(L, I_col, lower = True)
        
        for j in range(dim):
            UINV[j][columna] = UINV_col[j][0]
            LINV[j][columna] = LINV_col[j][0]

        I_col[columna][0] = 0
        
    return UINV@LINV


C = inversaLU(B[0], B[1])

print(A@C)
