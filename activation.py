"""
Luiz Souza Neto - 15473503

André Tanikawa de Oliveira - 15048766

Pedro Teixeira Guimarães - 12821669

Gustavo Jun Arakaki - 14028786

Renan Biruel Uema - 15486661

Renan Mochizuki - 15579831
"""

"""
activation.py

Implementação das funções de ativação utilizadas pela Rede Neural Perceptron Multicamadas (MLP).

Neste projeto é utilizada a função Sigmoide, responsável por introduzir não linearidade na rede neural e permitir o aprendizado de padrões mais complexos.
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

    A função sigmoide comprime qualquer valor para
    o intervalo entre 0 e 1.
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

    Como a saída da sigmoide já está disponível durante o processo de backpropagation, utiliza-se a forma:

        x * (1 - x)

    onde x representa o valor já calculado pela sigmoide.
    """
    return x * (1.0 - x)