"""
Luiz Souza Neto - 15473503

André Tanikawa de Oliveira - 15048766

Pedro Teixeira Guimarães - 12821669

Gustavo Jun Arakaki - 14028786

Renan Biruel Uema - 15486661

Renan Mochizuki - 15579831
"""

"""
dataset.py

Responsável por carregar e preparar os datasets
utilizados pela MLP.
"""

from pathlib import Path
import numpy as np

# Diretório onde estão armazenados todos os datasets utilizados
DATA_DIR = Path("data")


# ==========================================================
# FUNÇÕES AUXILIARES
# ==========================================================

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

    Nota: o embaralhamento evita viés na divisão dos dados.
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
    # pois a sigmoide trabalha naturalmente com saídas entre 0 e 1.

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
    x_file="X.npy",
    y_file="Y_classe.npy"
):

    x_path = DATA_DIR / x_file
    y_path = DATA_DIR / y_file

    X = np.load(
        x_path,
        allow_pickle=True
    )

    Y = np.load(
        y_path,
        allow_pickle=True
    )

    # Cada caractere é armazenado originalmente como uma matriz
    # 10x12. Como a MLP trabalha com vetores de entrada,
    # cada amostra é transformada em um vetor de 120 atributos.
    #
    # Exemplo:
    #
    # (1326, 10, 12, 1)
    #        ↓
    # (1326, 120)
    #
    # onde 1326 representa o número de amostras.
    X = X.reshape(
        X.shape[0],
        -1
    ).astype(np.float64)

    if X.max() > 1:
        X = X / 255.0

    return X, Y, None


# ==========================================================
# FAUSETT
# ==========================================================

def load_fausett(filename):
    """
    Dataset Fausett

    Estrutura:
        63 entradas
         7 saídas
    """

    path = DATA_DIR / filename

    data = np.loadtxt(
        path,
        delimiter=",",
        encoding="utf-8-sig"
    )

    X = data[:, :63]

    Y = data[:, 63:]
    Y = np.where(
        Y == -1,
        0,
        1
    )

    return X, Y


def load_fausett_clean():
    return load_fausett("caracteres-limpo.csv")


def load_fausett_noise():
    return load_fausett("caracteres-ruido.csv")


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