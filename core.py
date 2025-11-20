from typing import List

import numpy as np
from numpy import int64
from numpy.typing import NDArray

from exceptions import TextoInvalidoError, MatrizInvalidaMultiplicacaoError

# Lista que mapeia cada letra do alfabeto com um número (índice da lista)
_LISTA_ALFABETO = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                   'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def _completar_texto(texto: str, n: int) -> str:
    '''
    Faz com que o tamanho do texto digitado seja
    um múltiplo de n. Faz isso adicionando 'X' ao
    fim do texto.

    Params
    ------
    texto : str
        Texto a ser completado.

    n : int
        Número inteiro do qual o tamanho do texto
        será múltiplo.

    Returns
    -------
    Texto completado com 'X' com um tamanho múltiplo de n.
    '''
    tamanho_texto = len(texto)

    qtd_caracteres_faltando = tamanho_texto % n

    string_x = ''.join(['X' for _ in range(qtd_caracteres_faltando)])

    return texto + string_x

def _converter_texto_em_numeros(texto: str) -> NDArray[int64]:
    '''
    Converte um texto em uma lista de inteiros, de forma
    que cada inteiro da lista seja numericamente equivalente
    à letra do texto inicial.

    Params
    ------
    texto : str
        Texto a ser convertido.

    Returns
    -------
    Lista de inteiros obtida a partir do texto.

    Raises
    ------
    TextoInvalidoError se um dos caracteres do texto estiver fora do grupo
    de A-Z (case insensitive).
    '''

    try:
        return np.array([_LISTA_ALFABETO.index(char.upper()) for char in texto])
    except ValueError as e:
        raise TextoInvalidoError(f"Um dos caracteres do texto '{texto}' é inválido!")

def _multiplicar_matrizes(matriz_a: List[int | float] | List[List[int | float]] | NDArray,
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

def _converter_numeros_em_texto(numeros: List[int] | List[List[int]] | NDArray[int64]) -> str:
    '''
    Converte uma lista ou matriz de números inteiros em uma string,
    de forma que cada letra da string seja equivalente numericamente
    ao número inicial.

    Params
    ------
    numeros : List[int], List[List[int]] ou NDArray[int64]
        Lista ou matriz de inteiros a ser convertida.

    Returns
    -------
    Texto obtido a partir dos números.
    '''

    texto: str = ''

    numeros_np = np.array(numeros).flatten() # converte a lista de números para numpy

    for numero in numeros_np:
        texto += _LISTA_ALFABETO[int(numero) % 26]

    return texto

def criptografar_hill(texto: str, matriz_codificadora: List[int] | List[List[int]] | NDArray[int64]) -> str:
    '''
    Encripta o texto passado como parâmetro
    utilizando a Cifra de Hill, utilizando
    a matriz passada como parâmetro.

    Params
    ------
    texto : str
        Texto a ser criptografado.

    matriz_codificadora : List[int], List[List[int]] ou NDArray[int64]
        Matriz que será utilizada para
        criptografar o texto.

    Returns
    -------
    Texto criptografado.
    '''
    texto_criptografado: str = ''

    matriz_codificadora_np = np.array(matriz_codificadora) # converte a matriz para numpy

    n = matriz_codificadora_np.shape[0] # ordem da matriz

    texto_multiplo_n = _completar_texto(texto, n) # completa o resto do texto com 'X'

    # Percorre o texto separando em grupos de n letras
    qtd_grupos = int( len(texto_multiplo_n) / n )

    for i in range(qtd_grupos):
        grupo_str = texto_multiplo_n[i*n : (i+1)*n] # obtém o grupo do texto

        grupo_int = _converter_texto_em_numeros(grupo_str) # converte em números

        grupo_multiplicado = _multiplicar_matrizes(matriz_codificadora_np, grupo_int) # multiplica a matriz codificadora pelo grupo

        grupo_multiplicado_str = _converter_numeros_em_texto(grupo_multiplicado)

        texto_criptografado +=grupo_multiplicado_str

    return texto_criptografado
