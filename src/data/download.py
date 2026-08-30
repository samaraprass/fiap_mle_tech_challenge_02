"""Módulo de download e ingestão de dados diretamente da UC Irvine API."""

from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo
from src.utils.logger import setup_logger

logger = setup_logger("data_downloader")

UCI_DATASET_ID = 468  # Online Shoppers Purchasing Intention Dataset


def download_dataset(output_path: Path) -> pd.DataFrame:
    """Baixa o dataset da UC Irvine e salva o CSV consolidado no destino.

    Args:
        output_path: Caminho completo para o arquivo CSV de saída.

    Returns:
        DataFrame consolidado com features e target.
    """
    logger.info(f"Buscando dataset ID {UCI_DATASET_ID} da UC Irvine...")
    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    features: pd.DataFrame = dataset.data.features
    targets: pd.DataFrame = dataset.data.targets

    df = pd.concat([features, targets], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        f"Dataset salvo com sucesso em: {output_path} ({len(df)} linhas, {df.shape[1]} colunas)"
    )
    return df


if __name__ == "__main__":
    from src.config import AppConfig

    config = AppConfig()
    download_dataset(config.raw_data_path)
