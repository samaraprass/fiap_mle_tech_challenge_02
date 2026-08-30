"""Utilitário de configuração e gerenciamento de logs estruturados."""

import logging
import sys


def setup_logger(name: str = "fiap_mle", level: int = logging.INFO) -> logging.Logger:
    """Configura e retorna um logger padronizado para a aplicação.

    Args:
        name: Nome do logger.
        level: Nível de severidade do logging.

    Returns:
        Instância configurada de logging.Logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
