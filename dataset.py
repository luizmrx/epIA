"""
dataset.py

Responsável por carregar e preparar os datasets
utilizados pela MLP.
"""

# Quando você executar pela primeira vez:

# X, Y, label_map = load_character_complete()
# print(X.shape)
# print(Y.shape)

# anote o resultado.

# Se aparecer algo como:

# (1326, 120)
# (1326, 26)

# então confirmamos definitivamente que a arquitetura do trabalho será:

# 120 → 30 → 26

# e podemos partir para o main.py praticamente sem mais dúvidas.

from pathlib import Path
import numpy as np


DATA_DIR = Path("data")


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

def one_hot_encode(labels):
    """
    Converte rótulos em vetores one-hot.

    Exemplo:

    A -> [1,0,0,...]
    B -> [0,1,0,...]
    """

    alphabet = sorted(list(set(labels)))

    label_to_index = {
        label: idx
        for idx, label in enumerate(alphabet)
    }

    num_classes = len(alphabet)

    Y = np.zeros((len(labels), num_classes))

    for i, label in enumerate(labels):
        Y[i, label_to_index[label]] = 1

    return Y, label_to_index


def train_validation_test_split(
    X,
    Y,
    train_ratio=0.70,
    validation_ratio=0.15,
    shuffle=True,
    random_seed=42
):
    """
    Divide os dados em:

    treino
    validação
    teste

    Exemplo:

    70% treino
    15% validação
    15% teste
    """

    if shuffle:
        np.random.seed(random_seed)

        indices = np.arange(len(X))
        np.random.shuffle(indices)

        X = X[indices]
        Y = Y[indices]

    total = len(X)

    train_end = int(total * train_ratio)
    validation_end = train_end + int(total * validation_ratio)

    X_train = X[:train_end]
    Y_train = Y[:train_end]

    X_validation = X[train_end:validation_end]
    Y_validation = Y[train_end:validation_end]

    X_test = X[validation_end:]
    Y_test = Y[validation_end:]

    return (
        X_train,
        Y_train,
        X_validation,
        Y_validation,
        X_test,
        Y_test
    )


# ==========================================================
# PORTAS LÓGICAS
# ==========================================================

def load_logic_dataset(filename):
    """
    Carrega datasets AND, OR e XOR.

    Formato:

    x1,x2,y
    """

    path = DATA_DIR / filename

    data = np.loadtxt(
        path,
        delimiter=","
    )

    X = data[:, :2]

    Y = data[:, 2].reshape(-1, 1)

    # Converte:
    # -1 -> 0
    #  1 -> 1

    Y = np.where(Y == -1, 0, 1)

    return X, Y


def load_and():
    return load_logic_dataset("problemAND.csv")


def load_or():
    return load_logic_dataset("problemOR.csv")


def load_xor():
    return load_logic_dataset("problemXOR.csv")


# ==========================================================
# CARACTERES COMPLETO
# ==========================================================

def load_character_complete(
    x_file="X.txt",
    y_file="Y_letra.txt"
):
    """
    Carrega o dataset CARACTERES COMPLETO.

    X.txt:
        atributos (120 pixels)

    Y_letra.txt:
        letras A-Z
    """

    x_path = DATA_DIR / x_file
    y_path = DATA_DIR / y_file

    X = np.loadtxt(
        x_path,
        delimiter=","
    )

    with open(y_path, "r", encoding="utf-8") as file:

        labels = [
            line.strip()
            for line in file
            if line.strip()
        ]

    if len(X) != len(labels):
        raise ValueError(
            f"Quantidade de amostras diferente: "
            f"X={len(X)} "
            f"Y={len(labels)}"
        )

    Y, label_map = one_hot_encode(labels)

    return X, Y, label_map


# ==========================================================
# FAUSETT
# ==========================================================

def load_fausett(filename):
    """
    Dataset de caracteres da Fausett.

    Assumimos que cada linha
    representa uma classe.

    21 linhas -> 21 classes
    """

    path = DATA_DIR / filename

    X = np.loadtxt(
        path,
        delimiter=","
    )

    num_classes = X.shape[0]

    Y = np.eye(num_classes)

    return X, Y


def load_fausett_clean():
    return load_fausett("caracteres_limpo.csv")


def load_fausett_noise():
    return load_fausett("caracteres_ruido.csv")


def load_fausett_noise20():
    return load_fausett("caracteres_ruido20.csv")


# ==========================================================
# INFORMAÇÕES DOS DATASETS
# ==========================================================

def dataset_info(X, Y):

    print("\n===== DATASET INFO =====")
    print(f"Amostras : {X.shape[0]}")
    print(f"Entradas : {X.shape[1]}")

    if len(Y.shape) > 1:
        print(f"Saídas   : {Y.shape[1]}")
    else:
        print("Saídas   : 1")

    print("========================\n")