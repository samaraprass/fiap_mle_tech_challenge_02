"""Módulo para promoção e registro de modelos no MLflow Model Registry."""

import mlflow
from mlflow.tracking import MlflowClient
from src.config import AppConfig
from src.utils.logger import setup_logger

logger = setup_logger("model_registry")


def register_best_model(
    experiment_name: str,
    metric_name: str = "pr_auc",
    registered_model_name: str = "online_shopper_intention_model",
) -> str:
    """Busca o melhor run pelo score da métrica e registra no Model Registry.

    Args:
        experiment_name: Nome do experimento.
        metric_name: Métrica usada para ordenação (ex: 'pr_auc', 'f1_score').
        registered_model_name: Nome do modelo registrado no Model Registry.

    Returns:
        URI do modelo registrado.
    """
    config = AppConfig()
    mlflow.set_tracking_uri(config.tracking_uri)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric_name} DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("Nenhum run encontrado no experimento especificado.")

    best_run = runs[0]
    best_run_id = best_run.info.run_id
    best_score = best_run.data.metrics.get(metric_name)

    logger.info(
        f"Melhor run identificada: {best_run_id} com {metric_name} = {best_score:.4f}"
    )

    model_uri = f"runs:/{best_run_id}/model"
    model_version = mlflow.register_model(
        model_uri=model_uri, name=registered_model_name
    )

    logger.info(
        f"Modelo registrado com sucesso: {registered_model_name} (Versão: {model_version.version})"
    )
    return model_uri


if __name__ == "__main__":
    app_config = AppConfig()
    exp_name = app_config.config["mlflow"]["experiment_name"]
    model_name = app_config.config["mlflow"]["registered_model_name"]
    register_best_model(experiment_name=exp_name, registered_model_name=model_name)
