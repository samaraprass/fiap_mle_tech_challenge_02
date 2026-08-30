"""Submódulo de ingestão e particionamento de dados."""

from src.data.download import download_dataset
from src.data.loader import load_raw_data, split_data

__all__ = ["download_dataset", "load_raw_data", "split_data"]
