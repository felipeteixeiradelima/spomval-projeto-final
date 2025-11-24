class MatrizInvalidaError(Exception):
    def __init__(self, message="Matriz inválida para esse tipo de operação"):
        self.message = message
        super().__init__(self.message)

class OrdemInvalidaError(Exception):
    def __init__(self, message, value=None):
        self.message = message
        self.value = value
        super().__init__(self.message)

class SistemaImpossivelError(Exception):
    def __init__(self, message="O sistema linear analisado é impossível"):
        self.message = message
        super().__init__(self.message)

class SistemaIndeterminadoError(Exception):
    def __init__(self, message="O sistema linear analisado é possível, porém indeterminado"):
        self.message = message
        super().__init__(self.message)