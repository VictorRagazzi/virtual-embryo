# Virtual Embryo Challenge – Temporal Task

Trabalho final da disciplina. Ataque à [Task 1 – Temporal](https://virtualembryo.ai/challenge/tasks/temporal) do Virtual Embryo Challenge (NeurIPS 2026).

## Objetivo

Prever a distribuição de expressão gênica do coração embrionário de camundongo em estágios futuros que o modelo nunca viu, dado os estágios anteriores.

| Split      | Estágio | Papel                                  |
|------------|---------|----------------------------------------|
| Treino     | E8.5    | Estágio real observado                 |
| Treino     | E9.5    | Estágio real observado                 |
| Validação  | E10.5   | Target — avaliado no leaderboard       |
| Teste      | E12.5   | Target final — oculto                  |

## Estrutura

```
virtualembryo/
├── data/                  # Arquivos .h5ad baixados da competição
├── src/
│   └── scripts/
│       └── visualize.py   # Exploração e visualização dos dados
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

```bash
# Instalar dependências com uv
uv sync

# Rodar visualização
uv run python src/scripts/visualize.py --input data/<arquivo>.h5ad
```

## Métricas de avaliação (Task 1)

| Métrica                   | Peso |
|---------------------------|------|
| Recuperação de genes DE   | 25%  |
| Direção da mudança        | 25%  |
| Distribuição de estados   | 30%  |
| Co-variação gene-gene     | 20%  |