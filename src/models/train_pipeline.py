"""Script executável do estágio de treinamento de modelo e tracking com MLflow (DVC)."""

import argparse
from pathlib import Path
import joblib
import mlflow
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import AppConfig, ROOT_DIR
from src.models.evaluate import evaluate_classifier
from src.models.train import create_model_pipeline
from src.utils.logger import setup_logger

logger = setup_logger("train_stage")


def run_training_pipeline(model_type: str = "xgboost") -> None:
    """Executa o treinamento do pipeline e persiste o modelo para o DVC e MLflow.

    Args:
        model_type: Nome do modelo ('xgboost', 'random_forest', 'logistic_regression').
    """
    config = AppConfig()
    target_col = config.config["data"]["target_column"]
    train_path = config.processed_dir / "train.csv"
    preprocessor_path = config.processed_dir / "preprocessor.joblib"

    logger.info(f"Carregando dados de treino de: {train_path}")
    train_df = pd.read_csv(train_path)
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    preprocessor = joblib.load(preprocessor_path)
    params = config.model_params.get(model_type, {})

    logger.info(f"Criando pipeline do modelo: {model_type} com hiperparâmetros: {params}")
    pipeline = create_model_pipeline(
        model_type=model_type,
        params=params,
        preprocessor=preprocessor,
    )

    # Configura MLflow Tracking
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment_name = config.config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    models_dir = ROOT_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_output_path = models_dir / "model.joblib"

    with mlflow.start_run(run_name=f"dvc_{model_type}_run"):
        logger.info("Ajustando pipeline no conjunto de treino...")
        pipeline.fit(X_train, y_train)

        # Salva o modelo treinado
        joblib.dump(pipeline, model_output_path)
        logger.info(f"Modelo treinado salvo em: {model_output_path}")

        # Registra parâmetros e modelo no MLflow
        mlflow.log_param("model_type", model_type)
        mlflow.log_params(params)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento de Modelo - Pipeline DVC")
    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "random_forest", "logistic_regression"],
        help="Tipo do estimador para treinar",
    )
    args = parser.parse_args()
    run_training_pipeline(model_type=args.model_type)
