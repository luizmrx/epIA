"""
plot_results.py

Geração dos gráficos utilizados no relatório.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path("results")

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
    accuracies = dados[:, 1]

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
    accuracies = dados[:, 1]

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

def gerar_todos_os_graficos():

    grafico_erro_epocas()

    grafico_acuracia_hidden_size()

    grafico_acuracia_learning_rate()

    print(
        "\nGráficos gerados com sucesso!"
    )