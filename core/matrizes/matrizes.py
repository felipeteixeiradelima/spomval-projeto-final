from typing import List

import numpy as np
from numpy import int64, float64
from numpy.typing import NDArray

from .exceptions import MatrizInvalidaMultiplicacaoError, MatrizInvalidaDeterminanteError, MatrizNaoInversivelModuloError

def converter_para_numpy(matriz: List[int] | List[List[int]] | NDArray[int64] | List[float] | List[List[float]] | NDArray[float64]) -> NDArray[int64] | NDArray[float64]:
    '''
    Converte matriz para array numpy.

    Params
    ------
    matriz : ListLike
        Matriz a ser convertida.

    Returns
    -------
    Matriz convertida em array numpy.
    '''
    matriz_np = np.array(matriz)

    n_linhas = matriz_np.shape[0]

    # Se é unidimensional, redimensiona a matriz
    if len(matriz_np.shape) < 2:
        matriz_np = matriz_np.reshape(n_linhas, 1)

    return matriz_np

def _is_matriz_quadrada(matriz: NDArray[int64] | NDArray[float64]) -> bool:
    '''
    Retorna True se a matriz passada como
    parâmetro é quadrada, senão retorna False.

    Params
    ------
    matriz: NDArray[int64] | NDArray[float64]
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
    matriz_a_np = converter_para_numpy(matriz_a)
    matriz_b_np = converter_para_numpy(matriz_b)

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

def determinante(matriz: List[int] | List[List[int]] | NDArray[int64] | List[float] | List[List[float]] | NDArray[float64]) -> float:
    """
    Calcula o determinante de uma matriz usando eliminação de Gauss
    com pivotamento parcial.

    Params
    ------
    matriz : NDArray[int64] | NDArray[float64]
        Matriz quadrada.

    Returns
    -------
    Determinante da matriz.

    Raises
    ------
    MatrizInvalidaMultiplicacaoError se as matrizes
    possuem dimensões incompatíveis.
    """
    matriz_np = converter_para_numpy(matriz)
    matriz_np = matriz_np.astype(float)       # Garante que a matriz seja float
    n = matriz_np.shape[0]

    # Verifica se matriz é quadrada
    if not _is_matriz_quadrada(matriz_np):
        raise MatrizInvalidaDeterminanteError("A matriz precisa ser quadrada, matriz recebida possui shape", matriz_np.shape)

    det = 1.0
    troca_sinal = 1

    for i in range(n):
        # Pivotamento parcial: escolher o maior elemento da coluna
        pivot = i + np.argmax(np.abs(matriz_np[i:, i]))

        # Se o pivot é zero → determinante = 0
        if matriz_np[pivot, i] == 0:
            return 0.0

        # Troca de linhas se necessário
        if pivot != i:
            matriz_np[[i, pivot]] = matriz_np[[pivot, i]]
            troca_sinal *= -1  # trocar linhas altera o sinal do determinante

        # Eliminação
        for j in range(i+1, n):
            fator = matriz_np[j, i] / matriz_np[i, i]
            matriz_np[j, i:] -= fator * matriz_np[i, i:]

    # O determinante é o produto dos elementos da diagonal × sinal das trocas
    elementos_diagonal = [matriz_np[i,i] for i in range(n)]

    produto_elementos_diagonal = 1

    for elemento in elementos_diagonal:
        produto_elementos_diagonal *= elemento

    det *= troca_sinal * produto_elementos_diagonal

    # Se for muito próximo de um inteiro, arredonda
    det_arredondada = round(det)

    if round(abs(det_arredondada - det), 10) == 0:
        return det_arredondada

    return det

def matriz_inversa(M):
    n = len(M)

    # Verificação básica: matriz deve ser quadrada
    for linha in M:
        if len(linha) != n:
            raise ValueError("A matriz deve ser quadrada (n x n).")

    # Cria matriz aumentada [M | I]
    # Faz cópia profunda para não alterar a original
    A = [linha[:] for linha in M]
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    aug = [A[i] + I[i] for i in range(n)]

    # Aplica eliminação de Gauss-Jordan
    for i in range(n):
        # Encontra pivô (se o pivô for zero, tenta trocar por linha abaixo)
        if aug[i][i] == 0:
            for k in range(i + 1, n):
                if aug[k][i] != 0:
                    aug[i], aug[k] = aug[k], aug[i]
                    break
            else:
                raise ValueError("A matriz não é inversível (determinante zero).")

        # Normaliza linha do pivô
        pivot = aug[i][i]
        for j in range(2 * n):
            aug[i][j] /= pivot

        # Elimina demais linhas
        for k in range(n):
            if k != i:
                fator = aug[k][i]
                for j in range(2 * n):
                    aug[k][j] -= fator * aug[i][j]

    # Extrai matriz inversa da parte direita
    inversa = [linha[n:] for linha in aug]
    return inversa

def obter_inverso_modular(matriz: NDArray[int64] | NDArray[float64], modulo: int) -> float:
    '''
    Obtém o inverso modular multiplicativo da
    matriz passada como parâmetro.

    Params
    ------
    matriz : NDArray[int64] | NDArray[float64]
        Matriz original.
    modulo : int
        Módulo que será utilizado no cálculo.

    Returns
    -------
    Inverso modular multiplicativo da matriz.

    Raises
    ------
    MatrizNaoInversivelModuloError se a matriz não for
    inversível pelo módulo dado.
    '''

    det = determinante(matriz)

    try:
        x = pow(det, -1, modulo) # type: ignore
        return x
    except ValueError:
        raise MatrizNaoInversivelModuloError(f"A matriz não é inversível módulo {modulo}.")

def multiplicar_matriz_por_escalar(matriz: NDArray[int64] | NDArray[float64], escalar: int | float):
    '''
    Faz a multiplicação da matriz
    passada como parâmetro pelo
    escalar passado como parâmetro.

    Params
    ------
    matriz : NDArray[int64] | NDArray[float64]
        Matriz a ser multiplicada pelo escalar.
    escalar: int | float
        Número racional para multiplicar a matriz por.

    Returns
    -------
    Matriz multiplicada pelo escalar.
    '''
    matriz_copia = matriz.copy()

    n_linhas = matriz_copia.shape[0]
    n_colunas = matriz_copia.shape[1]

    for i in range(n_linhas):
        for j in range(n_colunas):
            matriz_copia[i,j] = matriz_copia[i,j] * escalar

    return matriz_copia

def calcular_modulo_elementos_matriz(matriz: NDArray[int64] | NDArray[float64], modulo: int | float):
    '''
    Calcula matriz módulo modulo.

    Params
    ------
    matriz : NDArray[int64] | NDArray[float64]
        Matriz a ser operada.
    modulo: int | float
        Número racional a ser usado como módulo.

    Returns
    -------
    Matriz módulo modulo.
    '''
    matriz_copia = matriz.copy()

    n_linhas = matriz_copia.shape[0]
    n_colunas = matriz_copia.shape[1]

    for i in range(n_linhas):
        for j in range(n_colunas):
            matriz_copia[i,j] = matriz_copia[i,j] % modulo

    return matriz_copia

def matriz_cofatores(matriz : NDArray[int64] | NDArray[float64]):
    """
    Constrói a matriz de cofatores de uma matriz quadrada.

    Params
    ------
    matriz : NDArray[int64] | NDArray[float64]
        Matriz quadrada (n, n).

    Returns
    -------
    Matriz de cofatores C, onde
    C[i][j] = (-1)^(i+j) * det(submatriz(i,j)).
    """
    matriz_py = matriz.tolist()
    n = len(matriz_py)
    matriz_de_cofatores = [[0] * n for _ in range(n)]

    for linha in range(n):
        for coluna in range(n):

            # Submatriz removendo linha i e coluna j
            submatriz = (
                matriz_py[:linha] + matriz_py[linha+1:]  # remove linha
            )
            submatriz = [
                linha_atual[:coluna] + linha_atual[coluna+1:]
                for linha_atual in submatriz
            ]

            menor_complementar = determinante(submatriz) # type: ignore
            sinal = (-1) ** (linha + coluna)

            matriz_de_cofatores[linha][coluna] = sinal * menor_complementar

    return converter_para_numpy(matriz_de_cofatores)


def transposta(matriz : NDArray[int64] | NDArray[float64]) -> NDArray[int64] | NDArray[float64]:
    """
    Retorna a transposta de uma matriz M.

    Params
    ------
    matriz : NDArray[int64] | NDArray[float64]
        Matriz (n x m).

    Returns
    -------
    Matriz transposta (m x n).
    """
    matriz_py = matriz.tolist()

    return converter_para_numpy([list(coluna) for coluna in zip(*matriz_py)])


def matriz_adjunta(matriz: NDArray[int64] | NDArray[float64]) -> NDArray[int64] | NDArray[float64]:
    """
    Calcula a matriz adjunta (adj(A)) de uma matriz quadrada.

    Params
    ------
    matriz : NDArray[int64] | NDArray[float64]
        Matriz quadrada (n, n).

    Returns
    -------
    A matriz adjunta adj(A) = cof(A)^T
    """
    matriz_de_cofatores = matriz_cofatores(matriz)
    matriz_adjunta = transposta(matriz_de_cofatores)
    return matriz_adjunta
