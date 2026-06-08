"""
train.py

Funções auxiliares utilizadas durante o ciclo
completo de treinamento da Rede Neural MLP.

Responsabilidades:

- Treinamento da rede
- Avaliação de desempenho
- Cálculo de métricas
- Salvamento de pesos
- Salvamento do histórico de erro
- Registro dos hiperparâmetros utilizados
"""

import os
import numpy as np

# Diretório utilizado para armazenar todos os
# arquivos gerados durante os experimentos.
RESULTS_DIR = "results"


def create_results_folder():
    """
    Cria a pasta de resultados.
    """

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


# ==================================================
# PESOS
# ==================================================

def save_initial_weights(mlp):

    create_results_folder()

    np.savetxt(
        f"{RESULTS_DIR}/initial_W1.csv",
        mlp.W1,
        delimiter=","
    )

    np.savetxt(
        f"{RESULTS_DIR}/initial_W2.csv",
        mlp.W2,
        delimiter=","
    )

    np.savetxt(
        f"{RESULTS_DIR}/initial_b1.csv",
        mlp.b1,
        delimiter=","
    )

    np.savetxt(
        f"{RESULTS_DIR}/initial_b2.csv",
        mlp.b2,
        delimiter=","
    )


def save_final_weights(mlp):
    """
    Salva os pesos e bias após o treinamento.

    Os valores armazenados representam o modelo final
    aprendido pela rede neural.
    """

    create_results_folder()

    np.savetxt(
        f"{RESULTS_DIR}/final_W1.csv",
        mlp.W1,
        delimiter=","
    )

    np.savetxt(
        f"{RESULTS_DIR}/final_W2.csv",
        mlp.W2,
        delimiter=","
    )

    np.savetxt(
        f"{RESULTS_DIR}/final_b1.csv",
        mlp.b1,
        delimiter=","
    )

    np.savetxt(
        f"{RESULTS_DIR}/final_b2.csv",
        mlp.b2,
        delimiter=","
    )


# ==================================================
# HISTÓRICO DE ERRO
# ==================================================

def save_error_history(history):
    """
    Armazena o erro obtido em cada época.

    Esses dados podem ser utilizados posteriormente
    para construção de gráficos de aprendizagem.
    """

    create_results_folder()

    np.savetxt(
        f"{RESULTS_DIR}/error_history.csv",
        np.array(history),
        delimiter=","
    )


# ==================================================
# TREINAMENTO
# ==================================================

def train_network(
    mlp,
    X_train,
    Y_train,
    X_val=None,
    Y_val=None,
    epochs=1000,
    patience=50
):
    """
    Executa o processo completo de treinamento.

    Etapas:
    1. Salva os pesos iniciais
    2. Treina a rede
    3. Salva os pesos finais
    4. Salva o histórico de erro
    5. Registra os hiperparâmetros utilizados
    """

    save_initial_weights(mlp)

    history = mlp.train(
        X_train,
        Y_train,
        X_val=X_val,
        Y_val=Y_val,
        epochs=epochs,
        patience=patience
    )

    save_final_weights(mlp)
    save_error_history(history)
    save_hyperparameters(mlp)

    return history


# ==================================================
# ACURÁCIA
# ==================================================

def accuracy(y_true, y_pred):
    """
    Calcula a acurácia do modelo.

    A acurácia representa a proporção de amostras
    classificadas corretamente.
    """

    return np.mean(
        y_true == y_pred
    )


# ==================================================
# MATRIZ DE CONFUSÃO
# ==================================================

def confusion_matrix(
    y_true,
    y_pred,
    num_classes
):
    """
    Constrói a matriz de confusão.

    Linhas:
        classes reais

    Colunas:
        classes previstas

    A matriz permite analisar quais classes
    estão sendo confundidas pela rede.
    """

    matrix = np.zeros(
        (
            num_classes,
            num_classes
        ),
        dtype=int
    )

    for real, pred in zip(
        y_true,
        y_pred
    ):
        matrix[real][pred] += 1

    return matrix


# ==================================================
# TESTE
# ==================================================

def evaluate_network(
    mlp,
    X_test,
    Y_test
):
    """
    Avalia uma MLP já treinada.
    """

    predictions = mlp.predict_class(
        X_test
    )

    # Problemas com múltiplas classes utilizam
    # codificação one-hot.

    if Y_test.ndim > 1:

        true_labels = np.argmax(
            Y_test,
            axis=1
        )

        acc = accuracy(
            true_labels,
            predictions
        )

        cm = confusion_matrix(
            true_labels,
            predictions,
            Y_test.shape[1]
        )

    else:

        true_labels = Y_test.flatten()

        acc = accuracy(
            true_labels,
            predictions.flatten()
        )

        cm = None

    return {
        "accuracy": acc,
        "confusion_matrix": cm,
        "predictions": predictions
    }


# ==================================================
# SAÍDAS DO TESTE
# ==================================================

def save_predictions(
    predictions,
    filename="predictions.csv"
):

    create_results_folder()

    np.savetxt(
        f"{RESULTS_DIR}/{filename}",
        predictions,
        delimiter=",",
        fmt="%s"
    )


def save_confusion_matrix(matrix):

    if matrix is None:
        return

    create_results_folder()

    np.savetxt(
        f"{RESULTS_DIR}/confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d"
    )

def save_hyperparameters(
    mlp,
    filename=f"{RESULTS_DIR}/hyperparameters.txt"
):

    create_results_folder()
    with open(filename, "w", encoding="utf-8") as f:

        f.write("HIPERPARÂMETROS DA REDE\n")
        f.write("=" * 40 + "\n\n")

        f.write(
            f"Entradas: {mlp.input_size}\n"
        )

        f.write(
            f"Oculta: {mlp.hidden_size}\n"
        )

        f.write(
            f"Saídas: {mlp.output_size}\n"
        )

        f.write(
            f"Learning Rate: {mlp.learning_rate}\n"
        )

        f.write(
            "Inicialização: Xavier Uniforme\n"
        )