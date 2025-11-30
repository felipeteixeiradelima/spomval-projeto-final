from typing import List

import numpy as np
from numpy import int64
from numpy.typing import NDArray

from .exceptions import TextoInvalidoError

from . import matrizes

# Lista que mapeia cada letra do alfabeto com um número (índice da lista)
_LISTA_ALFABETO = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                   'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

_MODULO = 26

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

    qtd_caracteres_faltando = n - tamanho_texto % n

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
        raise TextoInvalidoError(f"Um dos caracteres do texto '{texto}' é inválido: ", e) from e

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
        texto += _LISTA_ALFABETO[int(numero) % _MODULO]

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

    matriz_codificadora_np = matrizes.converter_para_numpy(matriz_codificadora) # converte a matriz para numpy

    n = matriz_codificadora_np.shape[0] # ordem da matriz

    texto_multiplo_n = _completar_texto(texto, n) # completa o resto do texto com 'X'

    # Percorre o texto separando em grupos de n letras
    qtd_grupos = int( len(texto_multiplo_n) / n )

    for i in range(qtd_grupos):
        grupo_str = texto_multiplo_n[i*n : (i+1)*n] # obtém o grupo do texto

        grupo_int = _converter_texto_em_numeros(grupo_str) # converte em números

        grupo_multiplicado = matrizes.multiplicar_matrizes(matriz_codificadora_np, grupo_int) # multiplica a matriz codificadora pelo grupo

        grupo_multiplicado_modulo = matrizes.calcular_modulo_elementos_matriz(grupo_multiplicado, modulo=_MODULO)

        grupo_multiplicado_str = _converter_numeros_em_texto(grupo_multiplicado_modulo) # type: ignore

        texto_criptografado += grupo_multiplicado_str

    return texto_criptografado

def decriptografar_hill(texto_encriptografado: str,
                         matriz_codificadora: List[int] | List[List[int]] | NDArray[int64]) -> str:
    '''
    Decriptografa o texto encriptografado
    em cifra de hill usando a matriz codificadora.

    Params
    ------
    texto_encriptografado : str
        Texto a ser decriptografado.
    matriz_codificadora : List[int] | List[List[int]] | NDArray[int64])
        Matriz usada na criptografia do texto.

    Returns
    -------
    Texto descriptografado.
    '''

    # Converte matriz codificadora para numpy
    matriz_codificadora_np = matrizes.converter_para_numpy(matriz_codificadora)

    n = matriz_codificadora_np.shape[0]

    # Cria matriz decodificadora vazia
    matriz_decodificadora = np.empty(matriz_codificadora_np.shape)

    # Obtém matriz adjunta e inverso modular da matriz codificadora
    matriz_adjunta = matrizes.matriz_adjunta(matriz_codificadora_np)
    inverso_modular = matrizes.obter_inverso_modular(matriz_codificadora_np, modulo=_MODULO)

    # Obtém matriz decodificadora (inverso * matriz_adjunta % 26)
    matriz_decodificadora = matrizes.multiplicar_matriz_por_escalar(matriz_adjunta, inverso_modular)
    matriz_decodificadora = matrizes.calcular_modulo_elementos_matriz(matriz_decodificadora, modulo=_MODULO)

    # Itera sobre os elementos do texto de forma agrupada
    
    # Percorre o texto separando em grupos de n letras
    qtd_grupos = int( len(texto_encriptografado) / n )

    texto_decriptografado = ''

    for i in range(qtd_grupos):
        grupo_str = texto_encriptografado[i*n : (i+1)*n] # obtém o grupo do texto

        grupo_int = _converter_texto_em_numeros(grupo_str) # converte em números

        grupo_multiplicado = matrizes.multiplicar_matrizes(matriz_decodificadora, grupo_int) # multiplica a matriz decodificadora pelo grupo

        grupo_multiplicado_mod = matrizes.calcular_modulo_elementos_matriz(grupo_multiplicado, modulo=_MODULO)

        grupo_multiplicado_str = _converter_numeros_em_texto(grupo_multiplicado_mod) # type: ignore

        texto_decriptografado += grupo_multiplicado_str

    return texto_decriptografado

if __name__ == "__main__":
    matriz = [[2,1,2],[-1,4,1],[4,2,3]]
    teste = criptografar_hill("CONFIRMADO", matriz)

    print(teste)

    teste_convertido = decriptografar_hill(teste, matriz)

    print(teste_convertido)

