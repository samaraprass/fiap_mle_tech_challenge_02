# ==============================================================================
# Multi-stage Dockerfile para o Pipeline de Machine Learning
# FIAP / Pós Tech - Machine Learning Engineering (Fase 2)
# ==============================================================================

# ------------------------------------------------------------------------------
# Estágio 1: Builder (Instalação das dependências com Poetry)
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# Instala dependências de compilação essenciais
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instala o Poetry
RUN pip install --no-cache-dir "poetry==${POETRY_VERSION}"

# Copia arquivos de definição de dependências
COPY pyproject.toml poetry.lock* ./

# Instala dependências de produção dentro do ambiente virtual isolado (.venv)
RUN poetry install --only main --no-root

# ------------------------------------------------------------------------------
# Estágio 2: Runtime (Imagem final enxuta para execução)
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app" \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Instala Git (essencial para o DVC)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia o ambiente virtual criado no estágio builder
COPY --from=builder /app/.venv /app/.venv

# Configura o Git e inicializa o repositório no container para o DVC
RUN git config --global --add safe.directory /app \
    && git config --global user.email "mle@fiap.com" \
    && git config --global user.name "FIAP MLE" \
    && git init

# Copia o código-fonte, configurações e arquivos do projeto
COPY configs/ ./configs/
COPY src/ ./src/
COPY dvc.yaml ./
COPY .dvc/ ./.dvc/
COPY .dvcignore ./
COPY .env.example ./.env

# Cria diretórios necessários para persistência de artefatos
RUN mkdir -p data/raw data/processed models eval_plots mlruns

# Define o comando padrão para executar o pipeline completo do DVC
CMD ["dvc", "repro"]
