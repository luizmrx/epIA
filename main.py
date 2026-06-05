"""
main.py

Arquivo principal do projeto.

Executa:

1. Portas lógicas
2. Caracteres Fausett
3. Caracteres Completo
"""

from dataset import (
    load_and,
    load_or,
    load_xor,
    load_fausett_clean,
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

def test_fausett():

    print("\n" + "=" * 50)
    print("TESTANDO FAUSETT")
    print("=" * 50)

    X, Y = load_fausett_clean()
    print(X.shape)
    print(Y.shape)

    mlp = MLP(
        input_size=X.shape[1],
        hidden_size=15,
        output_size=Y.shape[1],
        learning_rate=0.1
    )

    train_network(
        mlp,
        X,
        Y,
        epochs=3000
    )

    predictions = mlp.predict_class(X)

    accuracy = (
        predictions ==
        range(len(predictions))
    ).mean()

    print(f"\nAcurácia: {accuracy:.4f}")


# ==================================================
# CARACTERES COMPLETO
# ==================================================

def test_character_complete():

    print("\n" + "=" * 50)
    print("TESTANDO CARACTERES COMPLETO")
    print("=" * 50)

    X, Y, _ = load_character_complete()
    print(X.shape)
    print(Y.shape)

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

    mlp = MLP(
        input_size=120,
        hidden_size=30,
        output_size=26,
        learning_rate=0.1
    )

    history = train_network(
        mlp,
        X_train,
        Y_train,
        epochs=3000
    )

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


# ==================================================
# MAIN
# ==================================================

def main():

    # --------------------------
    # Portas Lógicas
    # --------------------------

    test_logic_gate(
        "AND",
        load_and,
        hidden_size=2
    )

    test_logic_gate(
        "OR",
        load_or,
        hidden_size=2
    )

    test_logic_gate(
        "XOR",
        load_xor,
        hidden_size=4
    )

    # --------------------------
    # Fausett
    # --------------------------

    test_fausett()

    # --------------------------
    # Caracteres Completo
    # --------------------------

    test_character_complete()


if __name__ == "__main__":
    main()