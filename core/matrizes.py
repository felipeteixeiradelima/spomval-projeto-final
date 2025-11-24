from typing import List

import numpy as np
from numpy import int64, float64
from numpy.typing import NDArray

from exceptions import MatrizInvalidaMultiplicacaoError, MatrizInvalidaDeterminanteError, MatrizInvalidaInversaError

def _is_matriz_quadrada(matriz: NDArray[int64|float64]) -> bool:
    '''
    Retorna True se a matriz passada como
    parâmetro é quadrada, senão retorna False.

    Params
    ------
    matriz: NDArray[int64|float64]
        Matriz a ser validada.

    Returns
    -------
    True se matriz é quadrada, senão false.
    '''

    return matriz.shape[0] == matriz.shape[1]

def multiplicar_matrizes(matriz_a: List[int | float] | List[List[int | float]] | NDArray,
                         matriz_b: List[int | float] | List[List[int | float]] | NDArray) -> NDArray:
    '''
    Faz a multiplicação da matriz_a pela matriz_b.

    Params
    ------
    matriz_a : List[int | float], List[List[int | float]] ou NDArray
        Matriz a ser multiplicada.

    matriz_b : List[int | float], List[List[int | float]] ou NDArray
        Matriz a ser multiplicada.

    Returns
    -------
    Nova matriz resultante da multiplicação
    de matriz_a por matriz_b.

    Raises
    ------
    MatrizInvalidaMultiplicacaoError se as matrizes
    possuem dimensões incompatíveis.
    '''

    # Converte matrizes para numpy
    matriz_a_np = np.array(matriz_a)
    matriz_b_np = np.array(matriz_b)

    # Obtém dimensões das matrizes
    linhas_matriz_a, *colunas_matriz_a = matriz_a_np.shape
    linhas_matriz_b, *colunas_matriz_b = matriz_b_np.shape

    # Se a matriz for unidimensional, faz com que a quantidade de colunas seja 1
    if len(colunas_matriz_a) == 0:
        colunas_matriz_a = 1
        matriz_a_np = matriz_a_np.reshape(linhas_matriz_a, colunas_matriz_a)
    else:
        colunas_matriz_a = colunas_matriz_a[0]

    if len(colunas_matriz_b) == 0:
        colunas_matriz_b = 1
        matriz_b_np = matriz_b_np.reshape(linhas_matriz_b, colunas_matriz_b)
    else:
        colunas_matriz_b = colunas_matriz_b[0]

    # Se o número de colunas de A difere do número de linhas de B, ERRO!
    if colunas_matriz_a != linhas_matriz_b:
        raise MatrizInvalidaMultiplicacaoError("As matrizes possuem dimensões incompatíveis: "
                                               f"{matriz_a_np.shape} {matriz_b_np.shape}")

    matriz_resultante = np.zeros((linhas_matriz_a, colunas_matriz_b)) # inicia a matriz com 0

    for i in range(linhas_matriz_a):
        for j in range(colunas_matriz_b):
            for k in range(colunas_matriz_a):
                matriz_resultante[i,j] += matriz_a_np[i,k] * matriz_b_np[k,j]

    return matriz_resultante

def determinante(matriz: NDArray[int64|float64]) -> float:
    """
    Calcula o determinante de uma matriz usando eliminação de Gauss
    com pivotamento parcial.

    Params
    ------
    matriz : NDArray[int64|float64]
        Matriz quadrada.

    Returns
    -------
    Determinante da matriz.

    Raises
    ------
    MatrizInvalidaMultiplicacaoError se as matrizes
    possuem dimensões incompatíveis.
    """
    matriz = matriz.astype(float)       # Garante que a matriz seja float
    n = matriz.shape[0]

    # Verifica se matriz é quadrada
    if not _is_matriz_quadrada(matriz):
        raise MatrizInvalidaDeterminanteError("A matriz precisa ser quadrada, matriz recebida possui shape", matriz.shape)

    det = 1.0
    troca_sinal = 1

    for i in range(n):
        # Pivotamento parcial: escolher o maior elemento da coluna
        pivot = i + np.argmax(np.abs(matriz[i:, i]))

        # Se o pivot é zero → determinante = 0
        if matriz[pivot, i] == 0:
            return 0.0

        # Troca de linhas se necessário
        if pivot != i:
            matriz[[i, pivot]] = matriz[[pivot, i]]
            troca_sinal *= -1  # trocar linhas altera o sinal do determinante

        # Eliminação
        for j in range(i+1, n):
            fator = matriz[j, i] / matriz[i, i]
            matriz[j, i:] -= fator * matriz[i, i:]

    # O determinante é o produto dos elementos da diagonal × sinal das trocas
    elementos_diagonal = [matriz[i,i] for i in range(n)]

    produto_elementos_diagonal = 1

    for elemento in elementos_diagonal:
        produto_elementos_diagonal *= elemento

    det *= troca_sinal * produto_elementos_diagonal

    # Se for muito próximo de um inteiro, arredonda
    det_arredondada = round(det)

    if round(abs(det_arredondada - det), 10) == 0:
        return det_arredondada

    return det

def matriz_inversa(matriz: NDArray[int64|float64]) -> NDArray[float64]:
    '''
    Calcula a matriz inversa da matriz dada
    usando eliminação de Gauss-Jordan
    com pivotamento parcial

    Params
    ------
    matriz: NDArray[int64|float64]
        Matriz a ser invertida.

    Returns
    -------
    Matriz inversa.
    '''
    matriz = matriz.astype(float)       # Garante que a matriz seja float
    n = matriz.shape[0]

    # Verifica se matriz é quadrada
    if not _is_matriz_quadrada(matriz):
        raise MatrizInvalidaInversaError("A matriz precisa ser quadrada, matriz recebida possui shape", matriz.shape)

    matriz_list = matriz.tolist()

        # Cria cópia da matriz e identidade
    M = [linha[:] for linha in matriz_list]              # cópia profunda
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    # Monta a matriz aumentada [A | I]
    for i in range(n):
        M[i] += I[i]

    # Gauss-Jordan
    for i in range(n):
        # Pivotamento parcial: encontra linha com maior valor na coluna
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))

        # Se pivô é zero → matriz não é invertível
        if M[pivot][i] == 0:
            raise MatrizInvalidaInversaError("A matriz não é invertível (determinante zero).")

        # Troca de linhas
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]

        # Normaliza linha do pivô
        piv = M[i][i]
        M[i] = [x / piv for x in M[i]]

        # Zera os outros elementos na coluna
        for j in range(n):
            if j != i:
                fator = M[j][i]
                M[j] = [M[j][k] - fator * M[i][k] for k in range(2*n)]

    # Extrai a parte da direita como inversa
    inv = [linha[n:] for linha in M]

    inv_np = np.array(inv)

    return inv_np

def obter_inverso_modular(matriz: NDArray[int64|float64], modulo: int) -> float | None:
    '''
    Obtém o inverso modular multiplicativo da
    matriz passada como parâmetro.

    Params
    ------
    matriz : NDArray[int64|float64]
        Matriz original.
    modulo : int
        Módulo que será utilizado no cálculo.

    Returns
    -------
    Inverso modular multiplicativo da matriz.

    Raises
    ------
    ValueError se o inverso modular não existir.
    '''

    det = determinante(matriz)

    try:
        x = pow(det, -1, modulo) # type: ignore
        return x
    except ValueError:
        print(f"O inverso modular de {det} módulo {modulo} não existe.")
