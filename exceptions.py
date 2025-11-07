class TextoInvalidoError(Exception):
    '''
    Exception utilizada quando uma string prestes
    a ser convertida em números possui um caractere
    inválido.
    '''
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class MatrizInvalidaMultiplicacaoError(Exception):
    '''
    Exception utilizada quando duas matrizes prestes
    a serem multiplicadas não atendem às condições necessárias
    (num. linhas da matriz1 == num. colunas da matriz2).
    '''
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
