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
├── data/                              # Arquivos .h5ad baixados da competição
│   ├── E85.h5ad
│   ├── E95.h5ad
│   └── prediction_e10_5.h5ad          # Gerado pelo pipeline
├── figs/                              # Plots gerados (UMAP, distribuições, etc)
├── src/
│   ├── scripts/
│   │   ├── visualize.py               # Exploração e visualização dos dados brutos
│   │   ├── score_local.py             # Placar local da Task 1 (E10.5), igual ao leaderboard
│   │   └── get_dummy.py               # Gera uma submissão dummy para testar o formato
│   └── approaches/                    # Diferentes abordagens exploradas para a Task 1
│       ├── genomic_expression/        # Abordagem baseada em perfis de expressão gênica
│       ├── llm/                       # Abordagem usando LLM como preditor/prior
│       ├── statistical_approximation/ # PCA + delta de centroide (predict_e10_5.py)
│       └── unsupervised_learning/     # Clustering/embeddings (umap_vec.py e afins)
├── .env                                # Variáveis de ambiente (chave de API do LLM) — não versionar
├── pyproject.toml
├── uv.lock
└── README.md
```

> Nota: a divisão acima segue as pastas criadas em `src/approaches/`; ajuste os nomes dos scripts abaixo se a organização final for diferente.

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

## Score local (E10.5)

`src/scripts/score_local.py` calcula localmente o score da Task 1 na validação (E10.5), exatamente como o leaderboard, usando o E10.5 liberado pela organização. Foi verificado que reproduz 3 submissões reais do leaderboard com 4 casas decimais (por exemplo 58.2302, 52.5650 e 50.8304) e as 6 âncoras oficiais.

### Requisitos

- Arquivos oficiais em `data/`: `E95.h5ad` (17.057 células) e `E105.h5ad` (24.979 células, o E10.5 liberado).
- Dependências: `uv sync`, ou `pip install veckit anndata h5py numpy scipy scikit-learn`. O `veckit` é o pacote oficial do scorer no PyPI.

### Uso

Em Python, a partir da raiz do repositório:

```python
from src.scripts.score_local import score_local

resultado = score_local("data/E105.h5ad", "minha_previsao.h5ad")   # local_e95="data/E95.h5ad" por padrão
print(resultado["total"])
```

Pela linha de comando:

```bash
uv run python -m src.scripts.score_local --e105 data/E105.h5ad --e95 data/E95.h5ad --previsao minha_previsao.h5ad
```

Saída de exemplo:

```
Métrica         Peso   Skill (0-100)   Valor bruto
de_score        0.25         45.5642       -0.1648
de_direction    0.25         54.1944        0.1223
mmd_u           0.30         56.6695       0.06487
variogram       0.20         53.1227      0.004624
TOTAL                        52.5650
Nota: copy_last (copiar o E9.5) = 50 nesta escala.
```

O retorno é um dicionário com:

- `total`: score final (0 a 100);
- `componentes`: skill de 0 a 100 de cada métrica;
- `metricas_brutas`: valores brutos de `de_score`, `de_direction`, `mmd_u` e `variogram`;
- `info`: número de células usadas, avisos e pico de memória.

Escala: `total = 0.25·de_score + 0.25·de_direction + 0.30·mmd_u + 0.20·variogram`. Por construção, 50 corresponde a copy_last (copiar o E9.5) e 100 é o teto (metade do alvo contra a outra metade).

### Requisitos da previsão

O script valida o arquivo e mostra um erro claro quando algo não bate:

- entre 1.000 e 5.118 células;
- exatamente os 32.285 genes, na mesma ordem do E9.5 (`var_names`);
- valores finitos e não negativos;
- escala log1p CP10k (`sc.pp.normalize_total(adata, target_sum=1e4)` seguido de `sc.pp.log1p`). Se a escala parecer errada, é emitido apenas um aviso.

### Como funciona

O script replica a receita do servidor:

- subamostra de 10% de cada estágio (sorteio sem reposição, seed 0): o E9.5 gera 1.706 células de referência e o E10.5 gera 2.498 células, divididas ao meio (`split_half`, seed 0);
- a primeira metade (1.249 células) é o alvo, e a outra metade estima o teto;
- métricas oficiais do `veckit`, com as âncoras oficiais do T1:val (fontes: https://virtualembryo.ai/challenge/data e https://virtualembryo.ai/challenge/panels/index.json).

### Memória e tempo

Pico de cerca de 1,5 GB de RAM e 15 a 20 s por previsão. O script lê só as células sorteadas dos `.h5ad` grandes, então roda em máquinas modestas.

### Cuidado

O alvo tem apenas 1.249 células e é fixo. Otimizar demais contra esse score é sobreajustar a validação. O teste final (E12.5) é outro alvo, então este número não é estimativa da pontuação final.

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