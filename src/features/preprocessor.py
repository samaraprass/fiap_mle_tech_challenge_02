"""Módulo de construção dos pipelines de transformação e pré-processamento de features."""

from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(
    numerical_cols: List[str],
    categorical_cols: List[str],
    boolean_cols: List[str],
) -> ColumnTransformer:
    """Cria e retorna o ColumnTransformer para pré-processamento de features.

    Args:
        numerical_cols: Lista com os nomes das variáveis numéricas contínuas/discretas.
        categorical_cols: Lista com os nomes das variáveis categóricas.
        boolean_cols: Lista com os nomes das variáveis booleanas.

    Returns:
        Instância de ColumnTransformer configurada.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            ),
            ("bool", "passthrough", boolean_cols),
        ]
    )
