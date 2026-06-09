"""
plot_results.py

Geração dos gráficos utilizados no relatório.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path("results")

def arquivo_existe(caminho):
    return Path(caminho).exists()

def grafico_erro_epocas():

    arquivo = RESULTS_DIR / "error_history.csv"

    history = np.loadtxt(
        arquivo,
        delimiter=","
    )

    plt.figure(figsize=(10, 5))

    plt.plot(history)

    plt.title(
        "Erro durante o treinamento"
    )

    plt.xlabel("Épocas")
    plt.ylabel("MSE")

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "error_vs_epochs.png",
        dpi=300
    )

    plt.close()

def grafico_acuracia_hidden_size():

    arquivo = RESULTS_DIR / "hidden_size_results.csv"

    if not arquivo.exists():
        print(
            "Arquivo hidden_size_results.csv não encontrado."
        )
        return

    arquivo = (
        RESULTS_DIR /
        "hidden_size_results.csv"
    )

    dados = np.loadtxt(
        arquivo,
        delimiter=",",
        skiprows=1
    )

    hidden_sizes = dados[:, 0]
    accuracies = dados[:, 2]

    plt.figure(figsize=(10, 5))

    plt.plot(
        hidden_sizes,
        accuracies,
        marker="o"
    )

    plt.title(
        "Acurácia x Neurônios Ocultos"
    )

    plt.xlabel(
        "Quantidade de Neurônios"
    )

    plt.ylabel(
        "Acurácia"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "accuracy_vs_hidden.png",
        dpi=300
    )

    plt.close()

def grafico_acuracia_learning_rate():
    arquivo = RESULTS_DIR / "learning_rate_results.csv"

    if not arquivo.exists():
        print(
            "Arquivo learning_rate_results.csv não encontrado."
        )
        return

    arquivo = (
        RESULTS_DIR /
        "learning_rate_results.csv"
    )

    dados = np.loadtxt(
        arquivo,
        delimiter=",",
        skiprows=1
    )

    learning_rates = dados[:, 0]
    accuracies = dados[:, 2]

    plt.figure(figsize=(10, 5))

    plt.plot(
        learning_rates,
        accuracies,
        marker="o"
    )

    plt.title(
        "Acurácia x Learning Rate"
    )

    plt.xlabel(
        "Learning Rate"
    )

    plt.ylabel(
        "Acurácia"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR /
        "accuracy_vs_lr.png",
        dpi=300
    )

    plt.close()

def grafico_tempo_hidden_size():

    arquivo = RESULTS_DIR / "hidden_size_results.csv"

    if not arquivo.exists():
        print(
            "Arquivo hidden_size_results.csv não encontrado."
        )
        return

    dados = np.loadtxt(
        "results/hidden_size_results.csv",
        delimiter=",",
        skiprows=1
    )

    hidden = dados[:,0]
    tempo = dados[:,1]
    accuracies = dados[:,2]

    plt.figure(figsize=(10,5))

    plt.plot(
        hidden,
        tempo,
        marker="o"
    )

    plt.title(
        "Tempo de Treinamento x Neurônios Ocultos"
    )

    plt.xlabel(
        "Neurônios Ocultos"
    )

    plt.ylabel(
        "Tempo (s)"
    )

    plt.grid(True)

    plt.savefig(
        "results/time_vs_hidden.png",
        dpi=300
    )

    plt.close()

def grafico_tempo_learning_rate():

    arquivo = RESULTS_DIR / "learning_rate_results.csv"

    if not arquivo.exists():
        print(
            "Arquivo learning_rate_results.csv não encontrado."
        )
        return

    dados = np.loadtxt(
        "results/learning_rate_results.csv",
        delimiter=",",
        skiprows=1
    )

    lr = dados[:,0]
    tempo = dados[:,1]

    plt.figure(figsize=(10,5))

    plt.plot(
        lr,
        tempo,
        marker="o"
    )

    plt.title(
        "Tempo de Treinamento x Learning Rate"
    )

    plt.xlabel(
        "Learning Rate"
    )

    plt.ylabel(
        "Tempo (s)"
    )

    plt.grid(True)

    plt.savefig(
        "results/time_vs_lr.png",
        dpi=300
    )

    plt.close()

def grafico_tempo_epocas():

    arquivo = RESULTS_DIR / "epochs_time.csv"

    if not arquivo.exists():
        print(
            "Arquivo epochs_time.csv não encontrado."
        )
        return

    dados = np.loadtxt(
        "results/epochs_time.csv",
        delimiter=",",
        skiprows=1
    )

    epochs = dados[:,0]
    tempo = dados[:,1]

    plt.figure(figsize=(10,5))

    plt.plot(
        epochs,
        tempo,
        marker="o"
    )

    plt.title(
        "Tempo de Treinamento x Épocas"
    )

    plt.xlabel(
        "Épocas"
    )

    plt.ylabel(
        "Tempo (s)"
    )

    plt.grid(True)

    plt.savefig(
        "results/time_vs_epochs.png",
        dpi=300
    )

    plt.close()

def gerar_todos_os_graficos():

    grafico_erro_epocas()

    grafico_acuracia_hidden_size()

    grafico_acuracia_learning_rate()

    grafico_tempo_hidden_size()

    grafico_tempo_learning_rate()

    grafico_tempo_epocas()

    print(
        "\nTodos os gráficos foram gerados."
    )