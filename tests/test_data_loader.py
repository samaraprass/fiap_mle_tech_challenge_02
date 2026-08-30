"""Testes unitários para o carregador e particionador de dados."""

import pandas as pd
import pytest
from src.data.loader import split_data


def test_split_data_shapes_and_stratification():
    """Testa se a divisão estratificada mantém proporções corretas."""
    # Cria dataset sintético de teste
    df_dummy = pd.DataFrame({
        "feat1": range(100),
        "feat2": range(100, 200),
        "Revenue": [1] * 20 + [0] * 80,
    })

    X_train, X_test, y_train, y_test = split_data(
        df=df_dummy, target_column="Revenue", test_size=0.2, random_seed=42
    )

    assert len(X_train) == 80
    assert len(X_test) == 20
    assert y_train.sum() == 16  # 20% de 80 positivos = 16
    assert y_test.sum() == 4    # 20% de 20 positivos = 4
