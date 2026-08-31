"""Script executável do estágio de avaliação de modelos e geração de métricas DVC."""

import json
from pathlib import Path
import joblib
import mlflow
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src.config import AppConfig, ROOT_DIR
from src.models.evaluate import evaluate_classifier
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_stage")


def run_evaluation_pipeline() -> None:
    """Carrega o modelo salvo e o conjunto de teste para calcular e persistir as métricas."""
    config = AppConfig()
    target_col = config.config["data"]["target_column"]
    test_path = config.processed_dir / "test.csv"
    model_path = ROOT_DIR / "models" / "model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Arquivo do modelo não encontrado em: {model_path}")

    logger.info(f"Carregando conjunto de teste de: {test_path}")
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    logger.info(f"Carregando modelo treinado de: {model_path}")
    pipeline = joblib.load(model_path)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test.to_numpy(), y_pred, y_prob)

    # Salva metrics.json para o DVC
    metrics_path = ROOT_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Métricas DVC salvas em: {metrics_path}")

    # Salva pontos das curvas para DVC plots
    plots_dir = ROOT_DIR / "eval_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
    roc_df.to_csv(plots_dir / "roc_curve.csv", index=False)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_df = pd.DataFrame({"precision": precision, "recall": recall})
    pr_df.to_csv(plots_dir / "pr_curve.csv", index=False)

    # Log de métricas no MLflow
    mlflow.set_tracking_uri(config.tracking_uri)
    experiment_name = config.config["mlflow"]["experiment_name"]
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="dvc_evaluate_run"):
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(plots_dir / "roc_curve.csv"))
        mlflow.log_artifact(str(plots_dir / "pr_curve.csv"))

    logger.info(
        f"Avaliação concluída! ROC-AUC: {metrics['roc_auc']:.4f} | "
        f"PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f}"
    )


if __name__ == "__main__":
    run_evaluation_pipeline()
