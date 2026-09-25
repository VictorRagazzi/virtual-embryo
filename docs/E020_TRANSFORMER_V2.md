# E020 — Transformer temporal v2

## Relação com o scorer

O `veckit 0.1.2` calcula DE real com Mann–Whitney, Benjamini–Hochberg, limiar de efeito, ordenação discreta, conjuntos top-k e interseções. `de_direction` usa ranks SciPy e correlação parcial. `mmd_u` usa PCA ajustada na verdade e kernels completos amostrados. Essas operações não formam um caminho diferenciável e a loss abaixo é uma aproximação de treino, não uma reimplementação do score.

Para genes selecionados somente no treino, sejam `d_p = mean(X_pred)-mean(X_ref)` e `d_t = mean(X_target)-mean(X_ref)`. O rank suave é

```text
r_i(d) = 1 + sum_j sigmoid((d_j-d_i)/tau)
w_i = sigmoid((|d_t,i|-q_k(|d_t|))/tau)
L_DE = sum_i w_i smoothL1(d_p,i,d_t,i)/sum_i w_i
     + sum_i w_i (standardize(r_i(d_p))-standardize(r_i(d_t)))²/sum_i w_i
```

O cálculo é O(G²), porém blocado em `ranking_block_size`, sem materializar `G×G`; G é configurável (512 no smoke, 2.048 inicialmente no servidor). `tau` e `top_k` são configuráveis. A seleção híbrida usa 50% maior variância e 50% maior mudança temporal absoluta, remove duplicatas e completa pelos rankings seguintes; holdout e E10.5 nunca entram nela.

Para RFF-MMD, com frequências e fases fixadas pela seed:

```text
phi(x) = sqrt(2/F) cos(x W + b)
L_MMD = ||mean(phi(X_pred))-mean(phi(X_target))||²
```

O custo é O((N_pred+N_target)GF), sem matriz kernel entre células. Frequências não são treinadas. A geração de treino usa índices de resíduos pré-amostrados do último estágio observado, `z=mu_k+softplus(sigma_k)*residuo`, e somente as linhas do decoder congelado correspondentes aos genes selecionados.

## Busca e seleção

Fase 1 contém quatro arquiteturas 128/256 × 2/4 camadas, com 4/8 cabeças compatíveis e dropout 0,05. Fase 2 compara estrutural, +DE, +MMD e +DE+MMD, seguida de ajuste pequeno dos pesos e repetição da vencedora na seed 43; o orçamento pretendido é de no máximo 12 execuções completas. Não há MLP.

Para cada erro `m`, `ganho_m=(erro_copy_last-erro_modelo)/max(erro_copy_last,epsilon)`. O score é `0.40 ganho_DE + 0.30 ganho_MMD + 0.20 ganho_estrutura + 0.10 ganho_proporcao`. Early stopping maximiza esse score, preserva o melhor checkpoint e registra qualquer fallback de bloco após OOM.

O holdout D011 mede extrapolação temporal conhecida. As 2.500 células reservadas de D001/D002, estratificadas por `celltype`, medem apenas generalização populacional dentro dos mesmos estágios; o rótulo não entra no Transformer ou na inferência.

## Comandos portáveis

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_population_transformer_v2 \
  --config configs/population_transformer/smoke.yaml --smoke-test

# servidor, após copiar data/ e models/scGPT_heart/
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_population_transformer_v2 \
  --config configs/population_transformer/server.yaml --phase prepare
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_population_transformer_v2 \
  --config configs/population_transformer/server.yaml --phase search
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_population_transformer_v2 \
  --config configs/population_transformer/server.yaml --phase retrain
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_population_transformer_v2 \
  --config configs/population_transformer/server.yaml --phase generate

# ou todas as fases, com retomada automática dos runs já iniciados
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_population_transformer_v2 \
  --config configs/population_transformer/server.yaml --phase all
```

Cada run mantém `last.pt` (modelo, otimizador, época, histórico e early stopping) e `checkpoint.pt` (melhor score); nova execução retoma automaticamente `last.pt`. Caches e checkpoints ficam em `outputs/population_transformer/`, não em `/tmp`. `--phase prepare` regenera decoder, embeddings e tokens K=50 quando ausentes. Qualquer OOM deve ser registrado; a configuração lista blocos 128/64/32, sem alteração silenciosa.
