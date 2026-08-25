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

## Estratégia

Pipeline em três partes (`predict_e10_5.py`):

1. **PCA conjunto + delta de centroide** — ajusta um PCA (50 PCs) sobre E8.5+E9.5 concatenados, calcula o deslocamento de centroide no espaço PCA para cada cell type persistente entre os dois estágios, e aplica esse delta sobre as células de E9.5 para extrapolar E10.5. Tipos exclusivos de E9.5 são copiados sem alteração.
2. **LLM como prior biológico** — usa um LLM (few-shot com a transição real E8.5→E9.5) para propor quais cell types *novos* (por diferenciação) devem surgir em E10.5, em qual proporção e a partir de qual progenitor de E9.5 herdar o perfil de expressão.
3. **Montagem final** — combina tipos persistentes (extrapolados) + tipos exclusivos de E9.5 (copiados) + tipos novos (amostrados do progenitor indicado), sub-amostrando de forma estratificada até o total exigido pela submissão (1000–5118 células).

## Estrutura

```
virtualembryo/
├── data/                       # Arquivos .h5ad baixados da competição
│   ├── E85.h5ad
│   ├── E95.h5ad
│   └── prediction_e10_5.h5ad   # Gerado pelo pipeline
├── figs/                       # Plots gerados (UMAP, distribuições, etc)
├── src/
│   └── scripts/
│       ├── visualize.py        # Exploração e visualização dos dados brutos
│       ├── predict_e10_5.py    # Pipeline principal: PCA+delta + LLM + montagem
│       ├── umap_vec.py         # Reprodução do UMAP oficial (PCA → Harmony → UMAP)
│       └── get_dummy.py        # Gera uma submissão dummy (baseline zero) para testar o formato
├── .env                        # Variáveis de ambiente (chave de API do LLM) — não versionar
├── pyproject.toml
├── uv.lock
└── README.md
```

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
uv run python src/scripts/predict_e10_5.py \
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