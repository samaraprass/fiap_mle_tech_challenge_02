"""Módulo de carregamento e validação de configurações e variáveis de ambiente."""

import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

# Carrega variáveis de ambiente do .env
load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"
PARAMS_PATH = ROOT_DIR / "configs" / "model_params.yaml"


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Carrega um arquivo YAML e retorna um dicionário de configurações.

    Args:
        file_path: Caminho para o arquivo YAML.

    Returns:
        Dicionário com os dados carregados.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


class AppConfig:
    """Classe agregadora das configurações do projeto."""

    def __init__(self) -> None:
        self.config: Dict[str, Any] = load_yaml(CONFIG_PATH)
        self.model_params: Dict[str, Any] = load_yaml(PARAMS_PATH)
        self.tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        self.random_seed: int = int(os.getenv("RANDOM_SEED", self.config["project"]["seed"]))
        self.raw_data_path: Path = ROOT_DIR / self.config["data"]["raw_path"]
        self.processed_dir: Path = ROOT_DIR / self.config["data"]["processed_dir"]
