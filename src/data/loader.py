"""Módulo responsável pelo carregamento e particionamento dos dados brutos."""

from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(file_path: Path) -> pd.DataFrame:
    """Lê o arquivo CSV bruto e retorna um DataFrame do pandas.

    Args:
        file_path: Caminho completo para o arquivo CSV.

    Returns:
        DataFrame com os dados brutos.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado em: {file_path}")
    return pd.read_csv(file_path)


def split_data(
    df: pd.DataFrame,
    target_column: str = "Revenue",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa os dados em treino e teste com estratificação pelo target.

    Args:
        df: DataFrame com todas as variáveis.
        target_column: Nome da coluna alvo.
        test_size: Proporção do conjunto de teste.
        random_seed: Semente pseudoaleatória para reprodutibilidade.

    Returns:
        Tupla contendo (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    return train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
