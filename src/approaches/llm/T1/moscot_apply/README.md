# T1 - Pareamento via Optimal Transport (E8.5 -> E9.5)

Esta etapa implementa o **item 1** da abordagem: pareamento probabilístico de
células entre E8.5 e E9.5 via Optimal Transport (OT) **não-balanceado**, usando
[`moscot`](https://moscot.readthedocs.io/), sem restringir custo/pareamento por
`obs['celltype']`.

Os itens 2 (fine-tuning estilo perturb-GEP do scGPT) e 3 (aplicar o modelo em
E9.5 para gerar E10.5) ainda **não** estão implementados aqui - ficam para a
próxima etapa, consumindo o dataset pareado gerado por este módulo.

## Instalação (via uv)

```bash
uv add moscot scanpy anndata pandas numpy scipy
```

`moscot` traz `ott-jax` como backend de Sinkhorn (JAX). Em CPU funciona sem
configuração extra; se houver GPU disponível, o JAX a utiliza automaticamente
se instalado com suporte a CUDA.

## Uso

```bash
uv run python -m src.approaches.llm.T1.ot_pairing \
    --e85 data/E85.h5ad \
    --e95 data/E95.h5ad \
    --out-dir data \
    --epsilon 0.01 --tau-a 0.95 --tau-b 0.95 \
    --pairing-mode sample --n-samples 1
```

### Saídas (em `--out-dir`, default `data/`)

| Arquivo | Conteúdo |
|---|---|
| `e85_e95_ot_pairs.h5ad` | AnnData pareado: `X` = expressão da célula de E8.5, `layers['target']` = expressão da célula de E9.5 associada. `obs` traz `source_cell`, `target_cell`, `ot_weight`, `source_celltype`, `target_celltype`, etc. Este é o arquivo a ser consumido pelo fine-tuning (item 2). |
| `e85_e95_ot_pairs.csv` | Mesma tabela de pares em formato tabular simples (útil para inspeção/EDA). |
| `e85_e95_ot_transport.npz` | Matriz de transporte completa, esparsificada mantendo os top-k pesos por linha (`scipy.sparse.csr_matrix`), para reanálise sem precisar re-resolver o OT. |

## Como funciona

1. **Carregamento e alinhamento de genes** (`data_utils.load_timepoints`):
   E8.5 e E9.5 são restritos à interseção de genes (deve ser o total dos
   ~12.930 genes já filtrados por variância, se o filtro foi idêntico nos dois
   arquivos) e concatenados em um único `AnnData`, com `obs['time']` numérico
   (8.5 / 9.5), exigido pelo moscot.

2. **Embedding conjunto** (`data_utils.compute_joint_pca`): uma PCA é
   calculada **sobre os dois timepoints juntos** (não separadamente), para que
   a distância usada como custo do OT seja comparável entre E8.5 e E9.5. Isso
   fica em `obsm['X_pca_joint']` e não sobrescreve `X` (que permanece com a
   expressão log-normalizada, usada depois para montar o dataset pareado).

3. **Optimal Transport não-balanceado** (`ot_pairing.solve_ot`): usa
   `moscot.problems.time.TemporalProblem`, preparado apenas com
   `time_key='time'` e `joint_attr='X_pca_joint'` - **nenhum argumento de
   celltype é passado** a `.prepare()`/`.solve()`. `tau_a`/`tau_b < 1` tornam o
   OT não-balanceado (permitem que a massa de uma célula não seja totalmente
   conservada no acoplamento), o que é apropriado já que:
   - não há rastreamento real de linhagem entre os timepoints;
   - a composição/proporção de tipos celulares muda entre E8.5 e E9.5, e o
     transporte deve poder refletir isso livremente, incluindo mudança de tipo.

4. **Amostragem dos pares** (`pairing_sampler.transport_to_pairs`): a linha
   `i` da matriz de transporte (célula de E8.5) é normalizada e tratada como
   uma distribuição `P(alvo | fonte=i)`. Três modos:
   - `sample` (default): sorteia `--n-samples` alvos por célula fonte,
     proporcionalmente ao peso do OT (pareamento **probabilístico**, com
     `--n-samples 1` gera exatamente "uma célula associada por célula de
     E8.5", como pedido).
   - `argmax`: pega o alvo de maior peso (pareamento determinístico "melhor
     par").
   - `top_k`: mantém os `k` alvos de maior peso por célula fonte (sem
     aleatoriedade), útil para aumento de dados controlado.

   Células de E8.5 cuja soma de linha na matriz de transporte for desprezível
   (`--min-row-mass`, default `1e-8`) são descartadas do pareamento (o OT não
   encontrou nenhum alvo plausível em E9.5 para elas).

5. **Montagem do dataset pareado** (`build_dataset.build_paired_anndata`):
   gera o `AnnData` final com uma observação por par `(fonte, alvo)`.

## Parâmetros importantes

- `--epsilon`: regularização entrópica do Sinkhorn. Menor = pareamento mais
  "duro"/próximo do OT exato (mais lento/instável); maior = mais suave.
  Comece em `0.01` e ajuste observando `row_mass` e convergência.
- `--tau-a`, `--tau-b`: quanto mais baixos, mais "não-balanceado" o transporte
  (mais tolerância a criação/destruição de massa). `1.0` = balanceado.
- `--rank`: `-1` usa Sinkhorn de posto completo. Para os ~16k x 16k reais,
  considere um valor positivo (ex.: `100`) para acelerar e reduzir memória, ao
  custo de uma aproximação de baixo posto do acoplamento.
- `--skip-normalization`: use se `data/E85.h5ad`/`E95.h5ad` já estiverem
  normalizados/log-transformados (o filtro de variância mínima que vocês já
  aplicaram normalmente já pressupõe isso).

## Limitações conhecidas / próximos ajustes possíveis

- Não foram usados priors de proliferação/apoptose (`proliferation_key`/
  `apoptosis_key` do moscot) para informar as marginais do OT não-balanceado -
  as marginais são uniformes, com o desbalanceamento controlado só por
  `tau_a`/`tau_b`. Se houver genesets de ciclo celular/apoptose disponíveis,
  dá para incorporar via `moscot`'s `score_genes_for_marginals` para marginais
  mais informativas.
- A matriz de transporte completa (densa) pode ser grande em escala real
  (~16k x 16k ~ 1GB em float32); por isso ela é esparsificada (top-k por linha)
  antes de salvar em disco. A matriz densa em si fica disponível em memória
  durante a execução (`get_transport_matrix`), caso queira analisá-la sem
  reduzir antes de salvar.