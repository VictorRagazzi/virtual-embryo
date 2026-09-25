# Correções temporais simples — E024/E025

O método conserva o E021 inteiro. A única operação adicional é:

`predição = max(E021 + peso × (média_anterior − média_penúltima), 0)`

A soma ocorre apenas onde E021 já era positivo. Isso mantém zeros e não
envolve correspondência entre células. E025 limita a soma aos genes de maior
mudança absoluta nos dois estágios conhecidos. Não existe rede adicional.
Os dois estágios são E8.0/E8.25 no pseudo-holdout; E8.5 externo só é alvo de
avaliação. A correção não usa o alvo para escolher genes ou calcular deltas.

Ollama foi usado para docstring de operação NumPy, sem dados ou checkpoints.
A sugestão foi revisada e resumida. O teste local verifica não mutação,
preservação de zeros, clipping e float32. Não há dependência Ollama no método.

## Comandos

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e024_trend --baseline outputs/population_transformer/E021/fixed_eval_seed42 --output-dir outputs/population_transformer/E024/seed42_n512 --seed 42
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e024_trend --baseline outputs/population_transformer/E021/confirm_seed42_n1024 --output-dir outputs/population_transformer/E024/seed42_n1024 --weights 0.0625 --seed 42
for top in 250 1000; do
  UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e024_trend --experiment E025 --top-genes $top --weights 0.125 0.25 --baseline outputs/population_transformer/E021/fixed_eval_seed42 --output-dir outputs/population_transformer/E025/top${top}_seed42_n512 --seed 42
done
# Scorer para cada peso/configuração; exemplo:
UV_CACHE_DIR=/tmp/ve-uv-cache uv run veckit --task T1 --input outputs/population_transformer/E024/seed42_n1024/trend_0.0625.h5ad --target outputs/population_transformer/E021/confirm_seed42_n1024/target.h5ad --reference outputs/population_transformer/E021/confirm_seed42_n1024/reference.h5ad --seed 42 --out outputs/population_transformer/E024/seed42_n1024/veckit_0.0625.json
UV_CACHE_DIR=/tmp/ve-uv-cache uv run pytest tests/test_e024_trend.py tests/test_e023_programs.py tests/test_population_transformer_v2.py tests/test_population_pipeline_v2.py tests/test_population_temporal.py -q
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m compileall -q src/scripts/run_e024_trend.py tests/test_e024_trend.py
git diff --check
```

Para os demais scorers, substituir input/out e manter target/reference
`E021/fixed_eval_seed42/` para 512 ou `E021/confirm_seed42_n1024/` para 1.024.
Os seeds de programas e avaliação são sempre 42. A correção é determinística;
o seed do Transformer é o da população E021 usada como base.

## Limitações

O pseudo-holdout é reutilizado para seleção, não é teste independente. K50
herdado inclui o holdout. As métricas locais não garantem ganho no leaderboard.
E024 e E025 acrescentam uma tendência global, compartilhada pelos grupos;
isso é deliberadamente simples, mas pode confundir mudança de composição
com mudança de expressão dentro de um grupo.

Configurações, vetores de tendência e contratos com SHA-256/tamanhos ficam
nos diretórios de cada experimento. Checkpoint/32 programas/histórico são
os do E021 referenciado por `config.json`: não há novo treinamento.

## Confirmação e estabilidade E025

Top250/peso0,125 passou os gates de 1.024 na seed42: DE 0,1743,
direção 0,1199, MMD 0,06074, variograma 0,001265. A queda de direção
é de 4,92%, próxima do limite de 5%. Comandos adicionais:

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e024_trend --experiment E025 --top-genes 250 --weights 0.125 --baseline outputs/population_transformer/E021/confirm_seed42_n1024 --output-dir outputs/population_transformer/E025/top250_seed42_n1024 --seed 42
for model_seed in 43 44; do
  UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e021_anchored --experiment E025 --output-dir outputs/population_transformer/E025/base_seed${model_seed}_n1024 --n-programs 32 --n-cells 1024 --epochs 250 --patience 25 --seed $model_seed --program-seed 42 --evaluation-seed 42 --device cuda --scales 0.25 --proportion-scale 0.25
  UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e024_trend --experiment E025 --top-genes 250 --weights 0.125 --baseline outputs/population_transformer/E025/base_seed${model_seed}_n1024 --output-dir outputs/population_transformer/E025/top250_seed${model_seed}_n1024 --seed 42
done
for model_seed in 42 43 44; do
  UV_CACHE_DIR=/tmp/ve-uv-cache uv run veckit --task T1 --input outputs/population_transformer/E025/top250_seed${model_seed}_n1024/trend_0.125.h5ad --target outputs/population_transformer/E021/confirm_seed42_n1024/target.h5ad --reference outputs/population_transformer/E021/confirm_seed42_n1024/reference.h5ad --seed 42 --out outputs/population_transformer/E025/top250_seed${model_seed}_n1024/veckit_0.125.json
done
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.summarize_e025
```

Ollama também revisou os limites numéricos de top-k. Incorporada verificação
de faixa após revisão; rejeitada sugestão de corrigir valores inválidos
silenciosamente. Teste adicional verifica sinal, desempate determinístico e
erro explícito. Total atualizado: 20 testes focados passaram.
