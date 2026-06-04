"""
activation.py

Funções de ativação utilizadas pela rede neural MLP.
"""

import numpy as np


def sigmoid(x):
    """
    Função Sigmoide.

    Recebe:
        x -> valor escalar, vetor ou matriz

    Retorna:
        Valor transformado pela função sigmoide.

                    1
    sigmoid(x) = --------
                 1 + e^-x
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivada da função Sigmoide.

    Recebe:
        x -> saída da função sigmoide

    Retorna:
        Derivada da sigmoide.

    Fórmula:
        sigmoid'(x) = x * (1 - x)

    Observação:
        Esta implementação assume que o valor recebido
        já passou pela função sigmoide.
    """
    return x * (1.0 - x)