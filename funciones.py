import numpy as np
import scipy.linalg as sc

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

    Inv=np.zeros([dim,dim]).astype(float)
    
    I_col = np.zeros([dim,1]).astype(float)
    Inv_columnas=[]
    for columna in range(dim):
        
        I_col[columna][0]=1

        eta= sc.solve_triangular(L,I_col)
        Inv_col = sc.solve_triangular(U,eta)
        Inv_columnas.append(Inv_col)
        I_col[columna][0]=0
    
    for columna in range(dim):
        col_actual=Inv_columnas[columna]
        for fila in range(dim):
            Inv[fila][columna]=col_actual[fila][0]
    return Inv


C = inversaLU(B[0], B[1])

print(C)


#%%

import numpy as np
A=np.array([[1,4,7],
           [2,5,8],
           [3,6,10]])
def calcularLU(A):
    dim=A.shape[0]
    L=np.eye(dim)
    U=np.zeros([dim,dim]).astype(float)  
    
    multiplicadores=[] #Los utilizaremos para constuir L luego
    #Hago una copia de la Matriz A (para evitar aliasing)
    for fila in range(dim):
        for columna in range(dim):
            U[fila][columna]=A[fila][columna]

    #Vamos a triagunlar para que U sea triangular superior
    for fila in range(dim-1):
        for rep in range(1,dim-fila): #En cada fila tengo que triangular N=(Dim-fila) filas
            multiplicador=U[rep+fila][fila]/U[fila][fila] #Calculo los multiplicadores           
            multiplicadores.append(multiplicador)
            U[fila+rep]=U[fila+rep]-multiplicador*U[fila] #Triangulo las filas que estan por debajo de la actual
            
    #Ahora tenemos U, Consigamos L
    
    for fila in range(dim):
        for columna in range(dim):
            if (fila>columna): #Los elementos debajo de la diagonal
                L[fila][columna]=multiplicadores[fila+columna-1]
                
    return L,U
LU=calcularLU(A)    
print(LU[0]@LU[1])











