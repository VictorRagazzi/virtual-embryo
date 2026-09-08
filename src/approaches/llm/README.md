# Abordagens LLM – Virtual Embryo Challenge (Task 1: Temporal)

Este diretório contém duas abordagens distintas que utilizam **Modelos de Linguagem (LLMs)** e **Embeddings de Linguagem Biomédica** para a previsão da dinâmica temporal de expressão gênica do coração embrionário de camundongo ($E8.5 \to E9.5 \to E10.5$).

---

## 🧭 Visão Geral Comparativa

Em vez de tratar o scRNA-seq puramente como matrizes numéricas cegas, ambas as abordagens exploram o conhecimento biológico pré-treinado em modelos de linguagem, mas com paradigmas completamente diferentes:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                               src/approaches/llm/                                │
└──────────────────────────────────────────────────────────────────────────────────┘
         │                                                        │
         ▼                                                        ▼
┌───────────────────────────────────┐    ┌─────────────────────────────────────────┐
│     T1: Raciocínio Semântico      │    │     T2: Embeddings & Latent Velocity    │
│    Macro / Nível de População     │    │       Micro / Nível de Célula Única     │
├───────────────────────────────────┤    ├─────────────────────────────────────────┤
│ • Agrupa por tipo celular anotado │    │ • Serializa cada célula como texto      │
│ • LLM como especialista biológico │    │ • SentenceTransformer (PubMedBERT)      │
│ • Decide proporções e tendências  │    │ • kNN no espaço latente (RNA Velocity)  │
│ • Materialização determinística   │    │ • Decoder Ridge (Latente → Expressão)   │
└───────────────────────────────────┘    └─────────────────────────────────────────┘
```

| Característica | **Módulo T1 (Macro / Semântico)** | **Módulo T2 (Micro / Latente)** |
| :--- | :--- | :--- |
| **Nível de Abstração** | Nível de população / Tipo celular (*Cluster Cards*) | Célula a célula (*Single-cell textual serialization*) |
| **Tipo de LLM Utilizado** | LLM Generativo via Chat API (ex: DeepSeek, GPT-4, Claude) | Encoder Biomédico Local (`PubMedBERT-mnli-...`) |
| **Papel da IA** | Raciocínio biológico, proporções de linhagens e fatores de tendência ($\Delta$) | Mapeamento para espaço semântico latente contínuo de alta dimensão ($768\text{d}$) |
| **Extrapolação Temporal** | Fatores de tendência ($\text{trend\_factor}$) multiplicando deltas reais observados | Deslocamento vetorial kNN (*Latent Velocity*) entre $E8.5$ e $E9.5$ |
| **Geração de Expressão** | Determinística: amostragem de células reais de $E9.5$ com deltas e ruído | Decodificador supervisionado (`Ridge Regression`) treinado em $E8.5 + E9.5$ |
| **Linhagens Novas** | Suporta detecção/proposição explícita de `novel_lineage` pelo LLM | Extrapolação contínua no manifold latente |

---

# 🧬 Módulo T1: Planejamento Semântico e Materialização Determinística

O módulo **T1** foi projetado para contornar a principal limitação dos Modelos de Linguagem na biologia de célula única: **LLMs não são bons preditores de matrizes numéricas esparsas de altíssima dimensão** ($32.285 \text{ genes} \times 2.500 \text{ células}$). Pedir ao LLM que gere expressões brutas resulta em alucinação numérica descontrolada, quebra da distribuição de dropout e distorção de escala.

Para resolver isso, o T1 desacopla o problema em duas camadas complementares:
1. **Camada Macro-Semântica (LLM)**: Atua como biólogo especialista em desenvolvimento cardíaco embrionário de camundongo ($E8.5 \to E9.5 \to E10.5$), tomando decisões conceituais de alto nível (proporções celulares, tendências de aceleração/desaceleração de módulos biológicos e emergência de novas linhagens).
2. **Camada Micro-Estatística e Determinística (Algoritmos Numéricos)**: Constrói representações matriciais a partir dos dados observados reais ($E8.5$ e $E9.5$), projeta as decisões do LLM e sintetiza as novas células mantendo propriedades estatísticas, invariância de escala e covariância gênica.

---

## 🔄 Evolução do Paradigma no T1 ($v1 \to v2 \to v3$)

A arquitetura do T1 evoluiu a partir da análise das métricas da competição (Recuperação de Genes DE, Direção, Distribuição de Estados e Variograma de Covariância Gene-Gene):

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Evolução Arquitetural do T1                                 │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
  v1: Marcadores Isolados (Celltype vs Resto)
      └─ Problema: Marcadores de identidade não capturavam a dinâmica temporal dos genes (baixo DE score).
  
  v2: DE Temporal Pareado com Ruído Gaussiano Independente por Gene
      └─ Problema: Aplicar ruído N(0, σ²) gene a gene destrói a correlação biológica natural (variograma caiu de 49.2 para 45.2).
  
  v3 (Atual): Módulos de Genes Co-regulados (WGCNA-like) + PCA + Deslocamento Modular
      └─ Solução: Genes que variam juntos formam módulos. O LLM decide fatores de tendência por módulo.
         A materialização aplica um único ruído escalar por célula no módulo inteiro, preservando a covariância por construção!
```

---

## 🏗️ Arquitetura Detalhada do Pipeline (6 Estágios)

O pipeline integrado do T1 (`main.py`) opera em 6 estágios sequenciais com cache intermediário em disco:

```
  [data/E85.h5ad & data/E95.h5ad]
                │
                ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ ESTÁGIO 1: Descoberta Estatística de Módulos Co-regulados              │
  │ • Seleção de Top HVGs em E9.5 (Seurat)                                 │
  │ • Matriz de Correlação R e Distância D = 1 - R                         │
  │ • Agrupamento Hierárquico (Average Linkage) -> K Módulos               │
  │ • PCA 1D (Eigengene) por módulo -> Loadings (w) + Médias (μ)           │
  │ ──> data/gene_modules.json                                             │
  └────────────────────────────────────────────────────────────────────────┘
                │
                ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ ESTÁGIOS 2 & 3: Construção dos Cluster Cards Modulares                 │
  │ • Projeção dos scores das células em cada módulo: s = (x - μ) · w      │
  │ • Cálculo de delta_score por tipo celular (temporal vs população)     │
  │ • Extração dos top módulos mais alterados por tipo celular             │
  │ • Status populacional: persistent, new_in_e9.5, lost_after_e8.5       │
  │ ──> data/cluster_cards.json                                            │
  └────────────────────────────────────────────────────────────────────────┘
                │
                ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ ESTÁGIO 4: Raciocínio Semântico via LLM (OpenRouter / OpenAI API)      │
  │ • Prompting como especialista em cardiogênese murina                   │
  │ • Projeta proporções populacionais em E10.5 (target_proportion)        │
  │ • Define trend_factor por módulo (1.0 = linear, 0 = platô, <0 reverte) │
  │ • Propõe novel_lineage com parent_celltype e key_markers regulados    │
  └────────────────────────────────────────────────────────────────────────┘
                │
                ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ ESTÁGIO 5: Validação Pydantic Estrita & Auto-Correção                  │
  │ • Checagem de integridade de schema (tipos, módulos válidos, soma ~1.0)│
  │ • Loop com reenvio de traceback de erro ao LLM em caso de falha       │
  │ ──> data/transition_plan.json                                          │
  └────────────────────────────────────────────────────────────────────────┘
                │
                ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │ ESTÁGIO 6: Materialização Determinística com Preservação de Covariância │
  │ • Alocação exata de células via Algoritmo do Maior Resto (N=2500)      │
  │ • Amostragem do pool real de E9.5 (ou do progenitor parent_celltype)   │
  │ • Deslocamento por Módulo: Δ_mod = delta_score × trend_factor          │
  │ • Ruído Escalar por Célula: shift_cell = Δ_mod + N(0, |Δ_mod| × noise) │
  │ • Atualização Coordenada: x_novo = max(0, x_antigo + shift_cell × w)   │
  │ ──> predictions/prediction_e10_5_llm_decision.h5ad                     │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 Detalhamento Técnico dos Estágios

### Estágio 1: Descoberta de Módulos Co-regulados e Base Linear
1. **Seleção de HVGs**: Identifica os top-$N$ genes altamente variáveis (padrão: 2.000 genes) no estágio $E9.5$ via método `seurat` do Scanpy.
2. **Matriz de Distância de Co-expressão**: Calcula a matriz de correlação de Pearson $R \in \mathbb{R}^{N \times N}$ entre todos os HVGs. A distância de dissimilaridade é definida como:
   $$D_{ij} = 1 - R_{ij}$$
   Genes anti-correlacionados ($R = -1 \implies D = 2$) ficam com distância máxima e não são agrupados juntos.
3. **Agrupamento Hierárquico**: Aplica ligação média (*average linkage*) e divide o dendrograma em $K$ módulos discretos (padrão: 40 módulos), filtrando módulos com menos de `min_module_size` (8 genes).
4. **Base Eigengene (PCA 1D)**: Para cada módulo $m$, ajusta um PCA de 1 componente sobre as células de $E9.5$:
   - $\vec{\mu}_m$: vetor de média de expressão dos genes do módulo.
   - $\vec{w}_m$: vetor unitário de pesos (*loadings*) de cada gene ($\|\vec{w}_m\|_2 = 1$).
   - $\text{EVR}_m$: razão de variância explicada (*Explained Variance Ratio*).
5. **Score Modular de uma Célula**: A projeção de uma célula $c$ no módulo $m$ é um valor escalar contínuo:
   $$s_c^{(m)} = (\vec{x}_{c, m} - \vec{\mu}_m) \cdot \vec{w}_m$$

### Estágios 2 & 3: Cluster Cards Modulares por Tipo Celular
Para cada tipo celular presente no dataset:
- **Status Populacional**:
  - `persistent`: presente em $E8.5$ e $E9.5$.
  - `new_in_e9.5`: presente em $E9.5$, mas ausente em $E8.5$.
  - `lost_after_e8.5`: presente em $E8.5$, mas ausente em $E9.5$.
- **Cálculo do $\text{delta\_score}$ por Módulo**:
  - Para tipos `persistent`: avalia a mudança temporal interna:
    $$\Delta s^{(m)} = \bar{s}_{E9.5, \text{type}}^{(m)} - \bar{s}_{E8.5, \text{type}}^{(m)}$$
  - Para tipos `new_in_e9.5`: avalia a identidade do tipo frente ao tecido:
    $$\Delta s^{(m)} = \bar{s}_{E9.5, \text{type}}^{(m)} - \bar{s}_{E9.5, \text{população}}^{(m)}$$
- **Seleção dos Módulos Principais**: Ordena os módulos por $|\Delta s^{(m)}|$ e seleciona os top-$M$ (padrão: 10) para exibir no card, acompanhados de uma prévia dos genes com maiores pesos.

### Estágio 4: Raciocínio Semântico e Modelagem do LLM
O LLM recebe um resumo estruturado dos cards modulares e assume o papel de especialista em embriologia do coração de camundongo. Suas atribuições são:
- **`target_proportion`**: Estimar a fração populacional de cada tipo celular em $E10.5$ ($\sum \approx 1.0$).
- **`module_trends`**: Definir o `trend_factor` para cada módulo citado:
  - $\text{trend\_factor} = 1.0$: o módulo continuará mudando na mesma taxa observada entre $E8.5$ e $E9.5$.
  - $\text{trend\_factor} = 0.0$: a expressão do módulo atinge platô/estabilização.
  - $\text{trend\_factor} < 0.0$: reversão na dinâmica de regulação do módulo.
  - $\text{trend\_factor} > 1.0$: aceleração da diferenciação ou ativação.
  - Módulos não mencionados pelo LLM mantêm $\text{trend\_factor} = 1.0$ por padrão.
- **`novel_lineage` (Linhagens Novas)**: Propor tipos celulares que emergem entre $E9.5$ e $E10.5$ (ex: *Epicardium*, *Conduction System*, *Outflow Tract Cushion*), definindo:
  - `parent_celltype`: tipo progenitor obrigatório de onde derivam as células em $E9.5$ (ex: *Proepicardium*, *V-CM*, *NCC-derived*).
  - `key_markers`: lista de genes marcadores esperados, direção (`up` / `down`) e nível de confiança (`high`, `medium`, `low`).
- **`notes`**: Justificativa biológica concisa das decisões tomadas.

### Estágio 5: Validação Pydantic e Loop de Auto-Correção
O JSON retornado pelo modelo é estritamente validado com schemas Pydantic:
- Validação se a soma das proporções populacionais satisfaz $0.85 \le \sum \text{target\_proportion} \le 1.15$.
- Verificação contra alucinações: tipos celulares existentes e IDs de módulos devem obrigatoriamente constar nos cards de entrada.
- Validação de integridade para linhagens novas (`parent_celltype` válido e lista de `key_markers` não vazia).
- Em caso de falha, o sistema captura a exceção de validação e reenvia a mensagem de erro específica para o LLM corrigir em um loop de auto-correção de até $N$ tentativas (padrão: 3).

### Estágio 6: Materialização Estatística Determinística
1. **Alocação Exata de Células (Largest Remainder Method)**:
   - Converte as proporções contínuas $\text{target\_proportion}$ no número exato de células discretas por tipo celular para somar precisamente `--target-cells` (padrão: 2.500 células), evitando arredondamentos arbitrários.
2. **Amostragem do Pool Real de $E9.5$**:
   - Para tipos persistentes/continuantes: amostra células do mesmo tipo celular em $E9.5$.
   - Para linhagens novas: amostra células do tipo progenitor (`parent_celltype`) em $E9.5$.
3. **Cálculo dos Deslocamentos Modulares**:
   - Tipos conhecidos: $\Delta_{\text{mod}} = \Delta s^{(m)} \times \text{trend\_factor}$.
   - Linhagens novas: para marcadores que pertencem a um módulo, desloca o score do módulo proporcionalmente ao desvio padrão observado no pool progenitor ($\sigma_{\text{pool}}$) ponderado pela confiança:
     $$\Delta_{\text{mod}} = \text{sign} \times \sigma_{\text{pool}} \times \text{conf\_scale}$$
     (onde $\text{high} = 1.5, \text{medium} = 0.8, \text{low} = 0.3$). Se o gene não estiver em nenhum módulo, aplica deslocamento isolado (*fallback*).
4. **Preservação Matemática da Covariância**:
   - Para cada célula $c$ sorteada e para cada módulo $m$, sorteia um **único valor escalar de ruído**:
     $$\text{shift}_c^{(m)} = \Delta_{\text{mod}} + \mathcal{N}\left(0, |\Delta_{\text{mod}}| \times \text{noise\_frac}\right)$$
   - Atualiza todos os genes $g$ do módulo simultaneamente ao longo de seus pesos $\vec{w}_m$:
     $$x_{c, g}^{E10.5} = \max\left(0, \; x_{c, g}^{E9.5} + \text{shift}_c^{(m)} \times w_{m, g}\right)$$
   - Como todos os genes do módulo se movem coordenadamente pela base $\vec{w}_m$, a estrutura de co-variação biológica entre os genes é preservada com precisão.
5. **Geração do AnnData Final**:
   - Constrói o objeto `AnnData` com matriz esparsa CSR, preenchendo `obs['celltype']`, `obs['origin']`, barcodes padronizados (`E10.5-SYNTH-XXXXXX-1`) e variáveis `var` alinhadas a $E9.5$.

---

## 📁 Estrutura de Arquivos e Artefatos do Módulo T1

- [`main.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T1/main.py): Pipeline completo ponta a ponta (v3 - Eigengenes/Módulos de Genes Co-regulados).
- [`cluster_card.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T1/cluster_card.py): Script modular auxiliar (v1/v2 - Marcadores individuais `rank_genes_groups`).
- [`predict_transition.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T1/predict_transition.py): Script modular auxiliar (v1/v2 - Consulta LLM com base em marcadores individuais).
- `data/`: Diretório de cache dos artefatos intermediários (`gene_modules.json`, `cluster_cards.json`, `transition_plan.json`).

```
src/approaches/llm/T1/
├── main.py                  # Pipeline completo ponta a ponta (v3 - Eigengenes/Módulos)
├── cluster_card.py          # Script modular auxiliar (v1/v2 - Marcadores individuais)
├── predict_transition.py    # Script modular auxiliar (v1/v2 - Consulta LLM individual)
└── data/                    # Diretório de cache de artefatos intermediários
    ├── gene_modules.json    # Módulos, genes membros, loadings e médias
    ├── cluster_cards.json   # Cards estatísticos por tipo celular e módulo
    └── transition_plan.json # Plano de transição predito pelo LLM validado
```

### Exemplos dos Schemas JSON Intermediários

#### 1. `data/gene_modules.json` (Trecho)
```json
{
  "1": {
    "genes": ["Ttn", "Actc1", "Myl7", "Myl4", "Acta2", "Slc8a1", ...],
    "loadings": [0.234, 0.230, 0.226, 0.200, 0.198, 0.195, ...],
    "mean": [1.45, 2.10, 3.05, 1.88, ...],
    "explained_variance_ratio": 0.4821,
    "top_genes_preview": [
      { "gene": "Ttn", "loading": 0.234 },
      { "gene": "Actc1", "loading": 0.230 }
    ]
  }
}
```

#### 2. `data/cluster_cards.json` (Trecho)
```json
{
  "celltype_cards": [
    {
      "celltype": "AVC-CM",
      "status": "persistent",
      "n_cells_e8.5": 598,
      "n_cells_e9.5": 326,
      "proportion_e8.5": 0.03562,
      "proportion_e9.5": 0.01911,
      "top_modules": [
        {
          "module_id": 1,
          "basis": "temporal_within_type",
          "delta_score": -1.0542,
          "top_genes": [
            { "gene": "Ttn", "loading": 0.234 },
            { "gene": "Actc1", "loading": 0.230 }
          ]
        }
      ]
    }
  ]
}
```

#### 3. `data/transition_plan.json` (Trecho)
```json
{
  "predicted_celltypes": [
    {
      "celltype": "AVC-CM",
      "origin": "persistent",
      "parent_celltype": null,
      "target_proportion": 0.015,
      "module_trends": [
        { "module_id": 1, "trend_factor": 0.3 }
      ],
      "key_markers": []
    },
    {
      "celltype": "Epicardium",
      "origin": "novel_lineage",
      "parent_celltype": "Proepicardium",
      "target_proportion": 0.03,
      "module_trends": [],
      "key_markers": [
        { "gene": "Wt1", "direction": "up", "confidence": "high" },
        { "gene": "Tbx18", "direction": "up", "confidence": "high" }
      ]
    }
  ],
  "notes": "Downregulation progressiva do módulo sarcomérico em AVC-CM e emergência do epicárdio a partir do proepicárdio em E10.5."
}
```

---

## 🚀 Como Executar o Módulo T1

### Modo 1: Pipeline Completo Integrado (Recomendado - v3)

O script `main.py` executa o pipeline ponta a ponta com reaproveitamento inteligente de cache:

```bash
# Execução padrão (utiliza cache de módulos e LLM se já existirem)
uv run python src/approaches/llm/T1/main.py \
    --e85 data/E85.h5ad \
    --e95 data/E95.h5ad \
    --out predictions/prediction_e10_5_llm_decision.h5ad \
    --target-cells 2500 \
    --normalize

# Forçar o recálculo dos módulos e nova chamada ao LLM
uv run python src/approaches/llm/T1/main.py \
    --e85 data/E85.h5ad \
    --e95 data/E95.h5ad \
    --out predictions/prediction_e10_5_llm_decision.h5ad \
    --force-modules \
    --force-llm \
    --target-cells 2500
```

### Modo 2: Execução Modular Passo a Passo (v1 / v2)

Caso deseje executar os passos individualmente usando marcadores de expressão tradicionais:

```bash
# 1. Gerar os Cluster Cards por marcadores clássicos
uv run python src/approaches/llm/T1/cluster_card.py \
    --e85 data/E85.h5ad \
    --e95 data/E95.h5ad \
    --out src/approaches/llm/T1/data/cluster_cards.json \
    --top-n 15 \
    --normalize

# 2. Consultar o LLM para prever o plano de transição
uv run python src/approaches/llm/T1/predict_transition.py \
    --cards src/approaches/llm/T1/data/cluster_cards.json \
    --out src/approaches/llm/T1/data/transition_plan.json

# 3. Materializar as células E10.5
uv run python src/approaches/llm/T1/main.py \
    --e85 data/E85.h5ad \
    --e95 data/E95.h5ad \
    --out predictions/prediction_e10_5_llm_decision.h5ad \
    --target-cells 2500
```

---

## ⚙️ Tabela de Parâmetros e Flags do `main.py`

| Flag | Default | Descrição |
| :--- | :--- | :--- |
| `--e85` | *(Obrigatório)* | Caminho para o AnnData de treino do estágio $E8.5$ (`data/E85.h5ad`). |
| `--e95` | *(Obrigatório)* | Caminho para o AnnData de treino do estágio $E9.5$ (`data/E95.h5ad`). |
| `--out` | *(Obrigatório)* | Caminho para o AnnData sintético de predição $E10.5$ gerado. |
| `--target-cells` | `2500` | Número total exato de células sintéticas a serem materializadas. |
| `--n-hvgs` | `2000` | Quantidade de genes altamente variáveis selecionados para o clustering modular. |
| `--n-modules` | `40` | Número alvo de módulos co-regulados no clustering hierárquico. |
| `--min-module-size`| `8` | Tamanho mínimo de genes para um módulo ser considerado válido. |
| `--top-modules-per-type` | `10` | Quantidade de módulos com maiores deltas mostrados no prompt para cada tipo celular. |
| `--top-genes-shown` | `6` | Quantidade de genes mais representativos mostrados na prévia de cada módulo. |
| `--min-cells` | `20` | Número mínimo de células que um tipo celular deve ter em $E9.5$ para calcular deltas. |
| `--normalize` | `False` | Aplica `normalize_total(target_sum=1e4)` + `log1p` antes de extrair módulos e scores. |
| `--force-modules`| `False` | Ignora o cache de `gene_modules.json` e `cluster_cards.json` e recalcula do zero. |
| `--force-llm` | `False` | Ignora o cache de `transition_plan.json` e faz nova requisição à API do LLM. |
| `--model` | `$LLM_MODEL` | Modelo a ser consultado (ex: `deepseek/deepseek-v4-flash`, `gpt-4o`, etc.). |
| `--api-key` | `$OPENROUTER_API_KEY` | Chave de autenticação da API (carregada do `.env`). |
| `--base-url` | `https://openrouter.ai/api/v1` | URL base do endpoint compatível com a OpenAI API. |
| `--max-attempts`| `3` | Número de tentativas no loop de auto-correção em caso de erro no schema do LLM. |
| `--noise-frac` | `0.15` | Fração de desvio padrão do ruído gaussiano escalar aplicado por célula no módulo. |
| `--seed` | `0` | Semente pseudo-aleatória para amostragem determinística e reprodutibilidade. |

---

# 🔬 Módulo T2: Embeddings Biomédicos e RNA Velocity Latente

O módulo **T2** explora a geometria contínua do espaço latente gerado por encoders de linguagem biomédica (Sentence-Transformers). Cada célula é tratada individualmente como uma sentença textual descritiva de seu perfil de expressão.

```
       [E8.5 / E9.5 AnnData]
                 │
                 ▼  (1) embed.py  (PubMedBERT)
       [Embeddings 768d no Manifold]
                 │
                 ▼  (2) extrapolate.py  (kNN Latent Velocity)
       [Embeddings Extrapolados E10.5]
                 │
                 ▼  (3) reconstruct.py  (Ridge MultiOutput Decoder)
       [Expressão E10.5 (HVGs)]
                 │
                 ▼  (4) align_panel.py
       [prediction_e10_5_emb_aligned.h5ad (32.285 genes)]
```

### Arquivos e Responsabilidades

1. [`embed.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T2/embed.py):
   - Função `cell_to_text`: converte o vetor de expressão de uma célula em texto baseado nos $N$ genes mais expressos (ex: `"Mouse embryo cardiac cell gene expression: Sox17=3.20, Pou5f1=2.10..."`).
   - Utiliza o modelo `PubMedBERT-mnli-snli-scinli-scitail-mednli-stsb` via `sentence_transformers` para codificar as células em vetores de 768 dimensões normalizados no espaço esférico $L_2$.

2. [`extrapolate.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T2/extrapolate.py):
   - Implementa uma formulação análoga a **RNA Velocity no Espaço Latente**:
     - Para cada célula $i$ em $E9.5$, busca seus $k$ vizinhos mais próximos ($k=5$, distância cosseno) em $E8.5$.
     - Computa o vetor de deslocamento temporal:
       $$\vec{v}_i = \text{emb}(E9.5)_i - \frac{1}{k}\sum_{j \in \text{kNN}_{E8.5}} \text{emb}(E8.5)_j$$
     - Extrapola para o estágio futuro:
       $$\text{emb}(E10.5)_i = \text{emb}(E9.5)_i + \text{scale} \times \vec{v}_i$$
     - Normaliza o vetor resultante para manter a consistência no espaço métrico original.

3. [`reconstruct.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T2/reconstruct.py):
   - Treina um decodificador de regressão linear regularizada (`MultiOutputRegressor(Ridge(alpha=10.0))`) mapeando:
     $$\text{Embedding } (768\text{d}) \longrightarrow \text{Vetor de Expressão de HVGs}$$
   - Treinado nos pares reais de $E8.5 + E9.5$.
   - Aplica a transformação inversa nos embeddings extrapolados de $E10.5$, aplicando $\text{clip}(\cdot, 0, \infty)$.

4. [`align_panel.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T2/align_panel.py) & [`fix_panel.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T2/fix_panel.py):
   - Baixa e faz cache do painel oficial da competição (`T1__val.genes.txt`, 32.285 genes).
   - Reorganiza a matriz de expressão gerada para garantir alinhamento exato de colunas e ordem dos genes exigida pela validação, preenchendo genes não modelados com zero.

5. [`sanity_check.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T2/sanity_check.py):
   - Script de diagnóstico visual: gera projeção PCA 3D dos embeddings de $E8.5$ vs $E9.5$ para verificar se o encoder captura a separação biológica temporal e salva em `figs/sanity_check_llm_3d.png`.

6. [`main.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T2/main.py):
   - Pipeline de ponta a ponta que executa Pré-processamento $\to$ Embedding $\to$ Extrapolação $\to$ Reconstrução $\to$ Alinhamento ao Painel Oficial.

### Como Executar o Módulo T2

```bash
# Executar o pipeline completo T2
uv run python src/approaches/llm/T2/main.py

# (Opcional) Executar o sanity check 3D dos embeddings
uv run python src/approaches/llm/T2/sanity_check.py

# (Opcional) Corrigir/Realinhar um arquivo de predição existente ao painel oficial
uv run python src/approaches/llm/T2/fix_panel.py
```

---

## 📊 Síntese e Quando Usar Cada Módulo

- **Use T1 se:** Você quer incorporar **conhecimento ontológico e literatura biomédica profunda** sobre cardiogênese (ex: previsão de surgimento de epicárdio, septos ou sistema de condução cardíaca) mantendo controle estrito contra distorções numéricas.
- **Use T2 se:** Você quer modelar **trajetórias contínuas célula-a-célula no manifold**, aproveitando a topologia do espaço latente do PubMedBERT sem depender de anotações prévias de tipos celulares.

