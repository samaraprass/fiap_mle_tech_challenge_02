"""Script executável do estágio de pré-processamento e particionamento dos dados (DVC)."""

from pathlib import Path
import joblib
import pandas as pd
from src.config import AppConfig
from src.data.loader import load_raw_data, split_data
from src.features.preprocessor import build_preprocessor
from src.utils.logger import setup_logger

logger = setup_logger("preprocess_stage")


def run_preprocess_pipeline(config: AppConfig) -> None:
    """Executa o pipeline de divisão e pré-processamento de features.

    Args:
        config: Objeto com as configurações do projeto.
    """
    logger.info(f"Carregando dados brutos de: {config.raw_data_path}")
    df = load_raw_data(config.raw_data_path)

    cat_cols = config.config["features"]["categorical_features"]
    bool_cols = config.config["features"]["boolean_features"]
    num_cols = config.config["features"]["numerical_features"]
    target_col = config.config["data"]["target_column"]
    test_size = config.config["data"]["test_size"]
    seed = config.random_seed

    # Garante tipagem correta para o pré-processamento
    for col in cat_cols:
        df[col] = df[col].astype(str)
    for col in bool_cols:
        df[col] = df[col].astype(int)

    logger.info(f"Particionando dados: test_size={test_size}, seed={seed}")
    X_train, X_test, y_train, y_test = split_data(
        df=df, target_column=target_col, test_size=test_size, random_seed=seed
    )

    # Concatena novamente com o target para persistir train.csv e test.csv
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = config.processed_dir / "train.csv"
    test_path = config.processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logger.info(f"Conjuntos salvos: Treino={train_path} ({len(train_df)}), Teste={test_path} ({len(test_df)})")

    # Ajusta e salva o pré-processador ColumnTransformer
    preprocessor = build_preprocessor(
        numerical_cols=num_cols,
        categorical_cols=cat_cols,
        boolean_cols=bool_cols,
    )
    preprocessor.fit(X_train)

    preprocessor_path = config.processed_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"Pré-processador salvo com sucesso em: {preprocessor_path}")


if __name__ == "__main__":
    app_config = AppConfig()
    run_preprocess_pipeline(app_config)
