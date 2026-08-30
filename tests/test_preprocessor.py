"""Testes unitários para a pipeline de pré-processamento."""

import pandas as pd
from src.features.preprocessor import build_preprocessor


def test_build_preprocessor_transformation():
    """Valida se o ColumnTransformer executa as transformações sem erros."""
    df_sample = pd.DataFrame({
        "Administrative": [1.0, 2.0],
        "Administrative_Duration": [10.0, 20.0],
        "Month": ["Feb", "Mar"],
        "Weekend": [0, 1],
    })

    preprocessor = build_preprocessor(
        numerical_cols=["Administrative", "Administrative_Duration"],
        categorical_cols=["Month"],
        boolean_cols=["Weekend"],
    )

    transformed = preprocessor.fit_transform(df_sample)
    assert transformed is not None
    assert transformed.shape[0] == 2
