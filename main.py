"""
Luiz Souza Neto - 15473503

André Tanikawa de Oliveira - 15048766

Pedro Teixeira Guimarães - 12821669

Gustavo Jun Arakaki - 14028786

Renan Biruel Uema - 15486661

Renan Mochizuki - 15579831
"""

"""
main.py

Arquivo principal do projeto.

Executa:

1. Portas lógicas
2. Caracteres Fausett
3. Caracteres Completo
"""
import numpy as np

from dataset import (
    load_and,
    load_or,
    load_xor,
    load_fausett_clean,
    load_fausett_noise,
    load_fausett_noise20,
    load_character_complete,
    train_validation_test_split
)

from mlp import MLP

from train import (
    train_network,
    evaluate_network,
    save_predictions,
    save_confusion_matrix
)

from test_mesa import test_mesa


# ==================================================
# PORTAS LÓGICAS
# ==================================================

def test_logic_gate(
    name,
    dataset_loader,
    hidden_size
):

    print("\n" + "=" * 50)
    print(f"TESTANDO {name}")
    print("=" * 50)

    X, Y = dataset_loader()

    mlp = MLP(
        input_size=2,
        hidden_size=hidden_size,
        output_size=1,
        learning_rate=0.3
    )

    train_network(
        mlp,
        X,
        Y,
        epochs=5000
    )

    predictions = mlp.predict_class(X)

    print("\nEntradas:")
    print(X)

    print("\nEsperado:")
    print(Y.flatten())

    print("\nPrevisto:")
    print(predictions.flatten())


# ==================================================
# FAUSETT
# ==================================================

def test_hidden_sizes():
    """
    Avalia o impacto da quantidade de neurônios
    na camada oculta sobre a acurácia da rede.

    Objetivo:
    Determinar uma arquitetura adequada para
    o dataset de caracteres de Fausett.
    """

    print("\nTESTE DE NEURÔNIOS OCULTOS")

    X, Y = load_fausett_clean()

    for hidden in [5,10,15,20,25,30]:

        print(f"\nOculta = {hidden}")

        mlp = MLP(
            input_size=63,
            hidden_size=hidden,
            output_size=7,
            learning_rate=0.1
        )

        train_network(
            mlp,
            X,
            Y,
            epochs=3000
        )

        pred = mlp.predict_class(X)

        real = np.argmax(Y, axis=1)

        acc = np.mean(pred == real)

        print(f"Acurácia = {acc:.4f}")

def test_learning_rates():
    """
    Compara diferentes taxas de aprendizado.

    Objetivo:
    Identificar o valor que proporciona
    melhor convergência durante o treinamento.
"""

    print("\nTESTE DE LEARNING RATE")

    X, Y = load_fausett_clean()

    for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:

        print(f"\nLearning Rate = {lr}")

        mlp = MLP(
            input_size=63,
            hidden_size=25,
            output_size=7,
            learning_rate=lr
        )

        train_network(
            mlp,
            X,
            Y,
            epochs=3000
        )

        pred = mlp.predict_class(X)

        real = np.argmax(Y, axis=1)

        acc = np.mean(pred == real)

        print(f"Acurácia = {acc:.4f}")

def test_fausett():

    print("\n" + "=" * 50)
    print("TESTANDO FAUSETT")
    print("=" * 50)

    X, Y = load_fausett_clean()

    mlp = MLP(
        input_size=63,
        hidden_size=15,
        output_size=7,
        learning_rate=0.1
    )

    train_network(
        mlp,
        X,
        Y,
        epochs=3000
    )

    predictions = mlp.predict_class(X)

    true_labels = np.argmax(Y, axis=1)

    accuracy = np.mean(
        predictions == true_labels
    )

    print(f"\nAcurácia: {accuracy:.4f}")

def test_fausett_noise():

    print("\n" + "=" * 50)
    print("TESTE DE ROBUSTEZ AO RUÍDO")
    print("=" * 50)

    # Treinamento com dataset limpo
    X_train, Y_train = load_fausett_clean()

    mlp = MLP(
        input_size=63,
        hidden_size=15,
        output_size=7,
        learning_rate=0.1
    )

    train_network(
        mlp,
        X_train,
        Y_train,
        epochs=3000
    )

    # Teste com dataset ruidoso
    X_noise, Y_noise = load_fausett_noise()

    pred = mlp.predict_class(X_noise)

    real = np.argmax(Y_noise, axis=1)

    acc = np.mean(pred == real)

    print(f"Acurácia ruído = {acc:.4f}")

def test_fausett_noise20():

    print("\n" + "=" * 50)
    print("TESTE DE ROBUSTEZ AO RUÍDO 20")
    print("=" * 50)

    # Treinamento com dataset limpo
    X_train, Y_train = load_fausett_clean()

    mlp = MLP(
        input_size=63,
        hidden_size=15,
        output_size=7,
        learning_rate=0.1
    )

    train_network(
        mlp,
        X_train,
        Y_train,
        epochs=3000
    )

    # Teste com dataset ruidoso
    X_noise, Y_noise = load_fausett_noise20()

    pred = mlp.predict_class(X_noise)

    real = np.argmax(Y_noise, axis=1)

    acc = np.mean(pred == real)

    print(f"Acurácia ruído = {acc:.4f}")

# ==================================================
# CARACTERES COMPLETO
# ==================================================

def test_character_complete():

    print("\n" + "=" * 50)
    print("TESTANDO CARACTERES COMPLETO")
    print("=" * 50)

    X, Y, _ = load_character_complete()

    (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test
    ) = train_validation_test_split(
        X,
        Y,
        train_ratio=0.70,
        validation_ratio=0.15
    )

    print(f"\nTreino: {len(X_train)}")
    print(f"Validação: {len(X_val)}")
    print(f"Teste: {len(X_test)}")

    # Arquitetura final escolhida após os
    # experimentos de ajuste de hiperparâmetros.
    mlp = MLP(
        input_size=120,
        hidden_size=120,
        output_size=26,
        learning_rate=0.1
    )

    history = train_network(
        mlp,
        X_train,
        Y_train,
        X_val,
        Y_val,
        epochs=50000,
        patience=50
    )

    print("\nErro inicial:", history[0])
    print("Erro final:", history[-1])
    print("Épocas executadas:", len(history))

    results = evaluate_network(
        mlp,
        X_test,
        Y_test
    )

    print(
        f"\nAcurácia de teste: "
        f"{results['accuracy']:.4f}"
    )

    save_predictions(
        results["predictions"],
        filename="character_predictions.csv"
    )

    save_confusion_matrix(
        results["confusion_matrix"]
    )

    print(
        "\nMatriz de confusão salva."
    )

    print(
        "Predições salvas."
    )

def test_character_hidden_sizes():

    print("\n" + "=" * 60)
    print("TESTE DE NEURÔNIOS OCULTOS - CHARACTER COMPLETE")
    print("=" * 60)

    X, Y, _ = load_character_complete()

    (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test
    ) = train_validation_test_split(
        X,
        Y,
        train_ratio=0.70,
        validation_ratio=0.15
    )

    for hidden in [15, 30, 45, 60, 75, 90, 120, 150, 180, 210]:

        print(f"\nHidden Size = {hidden}")

        mlp = MLP(
            input_size=120,
            hidden_size=hidden,
            output_size=26,
            learning_rate=0.3
        )

        train_network(
            mlp,
            X_train,
            Y_train,
            epochs=10000
        )

        results = evaluate_network(
            mlp,
            X_test,
            Y_test
        )

        print(
            f"Acurácia = "
            f"{results['accuracy']:.4f}"
        )


def test_character_learning_rates():

    print("\n" + "=" * 60)
    print("TESTE DE LEARNING RATE - CHARACTER COMPLETE")
    print("=" * 60)

    X, Y, _ = load_character_complete()

    (
        X_train,
        Y_train,
        X_val,
        Y_val,
        X_test,
        Y_test
    ) = train_validation_test_split(
        X,
        Y,
        train_ratio=0.70,
        validation_ratio=0.15
    )

    for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:

        print(f"\nLearning Rate = {lr}")

        mlp = MLP(
            input_size=120,
            hidden_size=120,
            output_size=26,
            learning_rate=lr
        )

        train_network(
            mlp,
            X_train,
            Y_train,
            epochs=10000
        )

        results = evaluate_network(
            mlp,
            X_test,
            Y_test
        )

        print(
            f"Acurácia = "
            f"{results['accuracy']:.4f}"
        )

# ==================================================
# MAIN
# ==================================================

def main():

    # --------------------------
    # Validação matemática da implementação.
    #
    # Reproduz o exemplo apresentado no material
    # teórico para verificar Forward Propagation
    # e Backpropagation.
    # --------------------------

    # test_mesa()

    # --------------------------
    # Portas Lógicas
    # --------------------------

    # test_logic_gate(
    #     "AND",
    #     load_and,
    #     hidden_size=2
    # )

    # test_logic_gate(
    #     "OR",
    #     load_or,
    #     hidden_size=2
    # )

    # test_logic_gate(
    #     "XOR",
    #     load_xor,
    #     hidden_size=4
    # )

    # --------------------------
    # Fausett
    # --------------------------

    # test_hidden_sizes()
    # test_learning_rates()
    # test_fausett()
    # test_fausett_noise()
    # test_fausett_noise20()

    # --------------------------
    # Caracteres Completo
    # --------------------------

    test_character_complete()
    # test_character_hidden_sizes()
    # test_character_learning_rates()


if __name__ == "__main__":
    main()