"""Módulo de treinamento de modelos com integração ao MLflow Tracking."""

from typing import Any, Dict
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.features.preprocessor import build_preprocessor
from src.models.evaluate import evaluate_classifier
from src.utils.logger import setup_logger

logger = setup_logger("model_trainer")


def create_model_pipeline(
    model_type: str,
    params: Dict[str, Any],
    preprocessor: Any,
) -> Pipeline:
    """Instancia o pipeline Scikit-Learn combinando pré-processador e estimador.

    Args:
        model_type: Tipo do modelo ('logistic_regression', 'random_forest', 'xgboost').
        params: Dicionário de hiperparâmetros.
        preprocessor: Instância de ColumnTransformer configurada.

    Returns:
        Pipeline do Scikit-Learn.
    """
    if model_type == "logistic_regression":
        estimator = LogisticRegression(**params)
    elif model_type == "random_forest":
        estimator = RandomForestClassifier(**params)
    elif model_type == "xgboost":
        estimator = XGBClassifier(**params)
    else:
        raise ValueError(f"Tipo de modelo desconhecido: {model_type}")

    return Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])


def train_and_log_model(
    model_type: str,
    pipeline: Pipeline,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    params: Dict[str, Any],
    experiment_name: str,
) -> Dict[str, Any]:
    """Executa o treinamento, calcula métricas e registra a run no MLflow.

    Args:
        model_type: Nome descritivo do modelo.
        pipeline: Pipeline configurado para ajuste.
        X_train: Matriz de features de treino.
        y_train: Vetor alvo de treino.
        X_test: Matriz de features de teste.
        y_test: Vetor alvo de teste.
        params: Hiperparâmetros a serem logados.
        experiment_name: Nome do experimento no MLflow.

    Returns:
        Dicionário com as métricas de avaliação calculadas.
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_type}_run"):
        logger.info(f"Iniciando treinamento do modelo: {model_type}")
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_classifier(y_test, y_pred, y_prob)

        # Registro no MLflow
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        logger.info(
            f"Modelo {model_type} finalizado. ROC-AUC: {metrics['roc_auc']:.4f}, "
            f"PR-AUC: {metrics['pr_auc']:.4f}, F1: {metrics['f1_score']:.4f}"
        )
        return metrics
