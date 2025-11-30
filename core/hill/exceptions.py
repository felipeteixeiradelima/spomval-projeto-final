class TextoInvalidoError(Exception):
    '''
    Exception utilizada quando uma string prestes
    a ser convertida em números possui um caractere
    inválido.
    '''
    def __init__(self, *args: object) -> None:
        super().__init__(*args)
