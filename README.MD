# Virtual Embryo Challenge – Temporal Task

Trabalho final da disciplina. Ataque à [Task 1 – Temporal](https://virtualembryo.ai/challenge/tasks/temporal) do Virtual Embryo Challenge (NeurIPS 2026).

## Objetivo

Prever a distribuição de expressão gênica do coração embrionário de camundongo em estágios futuros que o modelo nunca viu, dado os estágios anteriores.

| Split      | Estágio | Papel                                  |
|------------|---------|-----------------------------------------|
| Treino     | E8.5    | Estágio real observado                 |
| Treino     | E9.5    | Estágio real observado                 |
| Validação  | E10.5   | Target — avaliado no leaderboard       |
| Teste      | E12.5   | Target final — oculto                  |

## Estrutura

```
virtualembryo/
├── data/                              # Arquivos .h5ad baixados da competição
│   ├── E85.h5ad
│   ├── E95.h5ad
│   └── prediction_e10_5.h5ad          # Gerado pelo pipeline
├── figs/                              # Plots gerados (UMAP, distribuições, etc)
├── models/
│   └── scGPT_heart/                   # Artefatos/modelos ligados ao uso de scGPT no domínio de coração
├── src/
│   ├── scripts/
│   │   ├── visualize.py               # Exploração e visualização dos dados brutos
│   │   ├── get_dummy.py               # Gera uma submissão dummy para testar o formato
│   │   └── pre_processement.py        # Filtro de baixa variância + agrupamento de genes correlacionados
│   └── approaches/                    # Diferentes abordagens exploradas para a Task 1
│       ├── genomic_expression/        # Abordagem baseada em perfis de expressão gênica
│       ├── llm/                       # Abordagens com modelos de linguagem/representação para expressão gênica
│       │   └── T1/
│       │       ├── fine_tuning/        # Experimentos de ajuste fino para adaptar o modelo à Task 1
│       │       └── moscot_apply/       # Aplicação de estratégia inspirada em MOSCOT/OT para mapear estados temporais
│       ├── statistical_approximation/ # PCA + delta de centroide (predict_e10_5.py)
│       └── unsupervised_learning/     # Clustering/embeddings (umap_vec.py e afins)
├── .env                                # Variáveis de ambiente (chave de API do LLM) — não versionar
├── pyproject.toml
├── uv.lock
└── README.md
```

> Nota: a divisão acima é uma visão de alto nível da organização do projeto. As descrições de `models/scGPT_heart/`, `src/approaches/llm/T1/fine_tuning/` e `src/approaches/llm/T1/moscot_apply/` são intencionais e resumidas, baseadas no papel indicado pelos nomes das pastas.

## Setup

```bash
# Instalar dependências com uv
uv sync

# Configurar a API key do provider de LLM usado na Parte 2 do pipeline
echo "OPENROUTER_API_KEY=sua_chave_aqui" >> .env

# (opcional) trocar o provider/modelo padrão
echo "LLM_PROVIDER=openrouter" >> .env
echo "LLM_MODEL=deepseek/deepseek-v4-flash" >> .env
```

## Comandos

### Explorar e visualizar os dados brutos

```bash
uv run python src/scripts/visualize.py --input data/E85.h5ad
uv run python src/scripts/visualize.py --input data/E95.h5ad --output figs/e95/
```

Use `--skip-preprocess` se o arquivo já tiver PCA/UMAP calculados no `.obsm`.

### Reproduzir o UMAP oficial (com correção de batch via Harmony)

```bash
uv run python src/scripts/umap_vec.py --h5ad data/E85.h5ad --out figs/umap_8.5.png

# Sem Harmony, para comparação
uv run python src/scripts/umap_vec.py --h5ad data/E85.h5ad --out figs/umap_8.5_no_harmony.png --no-harmony
```

### Rodar o pipeline de predição (E8.5 + E9.5 → E10.5)

```bash
uv run python src/scripts/predict_e10_5_PCA.py \
    --e85 data/E85.h5ad \
    --e95 data/E95.h5ad \
    --out data/prediction_e10_5.h5ad
```

Flags úteis:

| Flag              | Default | Descrição                                              |
|-------------------|---------|---------------------------------------------------------|
| `--n-comps`       | 50      | Número de componentes principais do PCA conjunto        |
| `--target-cells`  | 2500    | Total de células na predição final (faixa: 1000–5118)   |
| `--skip-llm`      | —       | Pula a Parte 2 (sem tipos novos), útil para debug rápido|
| `--seed`          | 0       | Seed para reprodutibilidade da amostragem               |

### Gerar uma submissão dummy (baseline para testar o formato de envio)

```bash
uv run python src/scripts/get_dummy.py
```

Gera `submission_dummy.h5ad`: mesma ordem de genes do dataset de referência, matriz esparsa toda zerada, 2500 células com barcodes sintéticos — serve só para validar o formato aceito pela plataforma antes de submeter a predição real.

### Pré-processar os dados de expressão gênica

```bash
uv run python src/scripts/pre_processement.py --e85 data/E85.h5ad --e95 data/E95.h5ad
```

Executa dois passos sobre os datasets E8.5 e E9.5 concatenados:

1. **Filtro de baixa variância** — remove genes com variância < `--variance-threshold` (default `0.01`) e salva a tabela de removidos em `data/preprocessing/joint_removed_genes.csv`.
2. **Agrupamento de genes correlacionados** — agrupa genes com correlação de Pearson ≥ `--correlation-threshold` (default `0.95`) via *union-find* em blocos, escolhe um representante por grupo (maior expressão média) e salva os grupos em `data/preprocessing/joint_gene_groups.csv`.

> **Pré-requisito:** os dados devem estar em escala log1p; o script detecta e normaliza automaticamente caso contrário.

Flags disponíveis:

| Flag                      | Default            | Descrição                                                   |
|---------------------------|--------------------|-------------------------------------------------------------|
| `--e85`                   | `data/E85.h5ad`    | Caminho para o dataset E8.5                                 |
| `--e95`                   | `data/E95.h5ad`    | Caminho para o dataset E9.5                                 |
| `--variance-threshold`    | `0.01`             | Variância mínima (log1p-space) para manter o gene           |
| `--correlation-threshold` | `0.95`             | Limiar de correlação de Pearson para agrupar dois genes     |
| `--output-dir`            | `data/preprocessing` | Pasta para salvar as tabelas CSV de saída                 |
| `--sample-cells`          | `5000`             | Células a amostrar por dataset (0 = usar todas)             |

Saídas geradas em `data/preprocessing/`:

| Arquivo                        | Conteúdo                                                         |
|-------------------------------|------------------------------------------------------------------|
| `joint_removed_genes.csv`     | Genes removidos: nome, variância, expressão média, % não-zero   |
| `joint_gene_groups.csv`       | Grupos: representante, membros, tamanho, correlação min/max      |



## Mouse-Geneformer multitemporal

A abordagem em [src/approaches/llm/T2](src/approaches/llm/T2/README.md) usa
Mouse-Geneformer pré-treinado com `t` e `Δt` explícitos e pareamento substituível
por vizinhança. O fluxo reserva 2.000 células de E8.5 e 2.000 de E9.5 antes do
treino e compara apenas essas amostras com `veckit`. O README da abordagem reúne
os comandos, recursos oficiais e limites do experimento.

## Métricas de avaliação (Task 1)

| Métrica                   | Peso |
|---------------------------|------|
| Recuperação de genes DE   | 25%  |
| Direção da mudança        | 25%  |
| Distribuição de estados   | 30%  |
| Co-variação gene-gene     | 20%  |

```python
from veckit import score

result = score(task="T1", input="data/E95.h5ad", target="data/E95.h5ad", reference="data/E85.h5ad")
print(result["metrics"])
```

--target (todas as tarefas) é o pseudo-alvo em relação ao qual a previsão de --input é avaliada. --reference (T1/T2) / --wt (T3) é a expressão de referência em relação à qual de_score/de_direction/severity_slope são calculados — essas são métricas PRIMÁRIAS (“você previu a mudança correta?”, e não apenas “isso parece plausível?”), portanto, passe um valor real sempre que estiver avaliando um modelo real; se omitido, o padrão é o próprio --input, o que só é correto quando você estiver testando deliberadamente uma linha de base sem alterações (copy_last/wt_identity) — nessas métricas, o valor exibido será corretamente 0, correspondendo às tabelas oficiais de linha de base.

Isso não é uma prévia da sua pontuação real na competição. O que quer que você passe como --target é, por definição, dados que você já possuía — portanto, isso apenas indica que o pipeline de pontuação funciona e que o formato da sua submissão é válido, não o quão bem você se sairá na meta real reservada.
