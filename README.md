# Tech Challenge - Fase 2 | FIAP Pós Tech (MLE)
> **Sistema Preditivo de Propensão de Compra em E-Commerce com MLOps**

Projeto desenvolvido para a Fase 2 do curso de **Machine Learning Engineering (FIAP / Pós Tech)**. O objetivo é construir um pipeline completo de Machine Learning focado em **Clean Code, reprodutibilidade com Poetry, versionamento de dados com DVC, containerização em Docker e rastreamento/registro de modelos via MLflow**.

---

## 📌 Visão Geral do Desafio

- **Problema de Negócio:** Identificar a propensão de compra (`Revenue`: `True`/`False`) de usuários a partir de seu comportamento de navegação no e-commerce.
- **Dataset:** *Online Shoppers Purchasing Intention Dataset* (12.330 sessões e 18 atributos).
- **Desafio Central:** Engenharia de Machine Learning, padrões de código, governança de dados e ciclo de vida do modelo.

---

## 🗂️ Estrutura do Repositório

```text
.
├── .dockerignore                    # Regras de exclusão do build Docker
├── .env.example                     # Template de variáveis de ambiente
├── .gitignore                       # Arquivos ignorados pelo Git (dados brutos DVC, caches, mlruns)
├── Dockerfile                       # Containerização do pipeline
├── dvc.yaml                         # Pipeline reprodutível do DVC
├── pyproject.toml                   # Gerenciador de dependências e metadados com Poetry
├── README.md                        # Documentação do projeto
│
├── configs/                         # Configurações centralizadas
│   ├── config.yaml                  # Parâmetros de dados, colunas e paths
│   └── model_params.yaml            # Hiperparâmetros dos modelos (LogReg, RF, XGBoost)
│
├── data/
│   ├── raw/                         # Dados brutos (versionados no DVC)
│   └── processed/                   # Dados pré-processados
│
├── notebooks/                       # Cadernos exploratórios
│   └── 01_eda_and_baseline_v0.ipynb # EDA detalhada + Benchmark v0 comparando 3 modelos
│
├── src/                             # Código-fonte modular de produção (Clean Code)
│   ├── __init__.py
│   ├── config.py                    # Leitura de configs e variáveis de ambiente
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                # Carga e particionamento estratificado
│   ├── features/
│   │   ├── __init__.py
│   │   └── preprocessor.py          # Pipeline Scikit-Learn (ColumnTransformer)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── evaluate.py              # Cálculo de métricas (PR-AUC, ROC-AUC, F1, etc.)
│   │   ├── registry.py              # Promoção no MLflow Model Registry
│   │   └── train.py                 # Treinamento e rastreamento no MLflow
│   └── utils/
│       ├── __init__.py
│       └── logger.py                # Logging estruturado
│
└── tests/                           # Testes unitários automatizados
    ├── __init__.py
    ├── test_data_loader.py
    └── test_preprocessor.py
```

---

## 🚀 Como Executar o Projeto

### 1. Clonar o Repositório e Configurar o Ambiente

```bash
# Copiar o arquivo de variáveis de ambiente
cp .env.example .env

# Instalar dependências via Poetry
poetry install

# Ativar o ambiente virtual do Poetry
poetry shell
```

### 2. Executar a Análise Exploratória e Benchmark v0 (Jupyter)

```bash
# Iniciar o Jupyter Lab ou Notebook
poetry run jupyter notebook notebooks/01_eda_and_baseline_v0.ipynb
```

---

## 📊 Benchmark v0 dos Modelos

A análise exploratória revelou um desbalanceamento severo de classes (~84.5% `False` vs ~15.5% `True`). Foram avaliados 3 modelos com ponderação de pesos de classe:

| Modelo | Estratégia de Balanceamento | ROC-AUC | PR-AUC (Avg Precision) | F1-Score (Classe 1) | Recall (Classe 1) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Regressão Logística** | `class_weight='balanced'` | ~0.89 | ~0.65 | ~0.61 | ~0.82 |
| **Random Forest** | `class_weight='balanced'` | ~0.93 | ~0.74 | ~0.67 | ~0.83 |
| **XGBoost Classifier** | `scale_pos_weight=5.46` | **~0.93** | **~0.75** | **~0.68** | **~0.84** |

> **Destaque:** `PageValues` é o preditor com maior peso e importância estatística para a conversão. XGBoost e Random Forest apresentaram o melhor equilíbrio entre Precision e Recall.
