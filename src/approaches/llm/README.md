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

O módulo **T1** aborda o desafio evitando que o LLM "alucine" diretamente dezenas de milhares de valores numéricos de expressão gênica. Em vez disso, o processo é particionado em **estágios macro-semânticos** e **estágios determinísticos**.

```
  [E8.5.h5ad & E9.5.h5ad]
             │
             ▼  (1) cluster_card.py
   [cluster_cards.json]  ── Proporções, status e deltas dos top marcadores
             │
             ▼  (2) predict_transition.py  <-- LLM (OpenRouter / OpenAI API)
  [transition_plan.json] ── Proporções E10.5, trend_factor, novel lineages
             │
             ▼  (3) main.py  <-- Determinístico
 [prediction_e10_5.h5ad] ── Células sintetizadas com base no pool real de E9.5
```

### Arquivos e Responsabilidades

1. [`cluster_card.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T1/cluster_card.py) *(Estágios 1–3)*:
   - Lê os dados de $E8.5$ e $E9.5$ anotados por tipo celular (`obs['celltype']`).
   - Identifica a dinâmica populacional: tipos `persistent`, `new_in_e9.5` ou `lost_after_e8.5`.
   - Executa `scanpy.tl.rank_genes_groups` (Wilcoxon/t-test) em $E9.5$ para identificar os top-$N$ genes marcadores por tipo celular.
   - Calcula as médias de expressão em $E8.5$, $E9.5$, o $\Delta(E9.5 - E8.5)$ e o $\log_2\text{FC}$ em relação ao restante do tecido.
   - Exporta um resumo compacto em JSON (`cards.json`), evitando passar matrizes gigantes ao LLM.

2. [`predict_transition.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T1/predict_transition.py) *(Estágios 4–5)*:
   - Constrói um prompt estruturado contendo as tabelas de tipos celulares e marcadores calculados.
   - Utiliza um System Prompt que instrui o LLM como **especialista em biologia do desenvolvimento cardíaco de camundongo**.
   - O LLM projeta para $E10.5$:
     - Proporções populacionais esperadas ($\sum \approx 1.0$).
     - `trend_factor`: multiplicador do delta observado ($1.0 =$ mantém velocidade, $0.0 =$ platô, negativo $=$ regressão, $>1 =$ aceleração).
     - `novel_lineage`: novas linhagens que emergem por diferenciação entre $E9.5$ e $E10.5$, com tipo celular progenitor (`parent_celltype`) e marcadores-chave esperados.
   - Validação estrita via **Pydantic** (`TransitionPlan`) com loop de auto-correção de até $N$ tentativas em caso de erro de schema ou alucinação de tipos inexistentes.

3. [`main.py`](file:///c:/Users/T.i/Desktop/trampo/Academico/Projeto_IA_GEN/Codar/virtual-embryo/src/approaches/llm/T1/main.py) *(Estágio 6 - Materialização)*:
   - Lê o `transition_plan.json`, os `cards.json` e o `E95.h5ad` original.
   - Aloca o número exato de células por tipo celular via algoritmo do **Maior Resto** (*Largest Remainder Method*) até totalizar `--target-cells`.
   - Amostra células reais da população de $E9.5$ (ou do tipo progenitor em caso de linhagem nova).
   - Aplica os deltas escalados:
     $$\Delta_{\text{aplicado}} = \Delta(E8.5 \to E9.5) \times \text{trend\_factor}$$
   - Para linhagens novas, modula os genes marcadores de acordo com o desvio padrão da população progenitora e a confiança biológica (`high`, `medium`, `low`).
   - Adiciona ruído gaussiano proporcional para evitar clones idênticos e salva o AnnData final (`.h5ad`).

### Como Executar o Módulo T1

```bash
# 1. Gerar os Cluster Cards a partir dos dados observados
uv run python src/approaches/llm/T1/cluster_card.py \
    --e85 data/E85.h5ad \
    --e95 data/E95.h5ad \
    --out src/approaches/llm/T1/data/cards.json \
    --top-n 15 \
    --normalize

# 2. Consultar o LLM para prever o plano de transição para E10.5
# (Requer OPENROUTER_API_KEY e LLM_MODEL definidos no .env)
uv run python src/approaches/llm/T1/predict_transition.py \
    --cards src/approaches/llm/T1/data/cards.json \
    --out src/approaches/llm/T1/data/transition_plan.json

# 3. Materializar a matriz de células sintéticas E10.5
uv run python src/approaches/llm/T1/main.py \
    --e95 data/E95.h5ad \
    --plan src/approaches/llm/T1/data/transition_plan.json \
    --cards src/approaches/llm/T1/data/cards.json \
    --target-cells 2500 \
    --out predictions/prediction_e10_5_llm_decision.h5ad
```

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

