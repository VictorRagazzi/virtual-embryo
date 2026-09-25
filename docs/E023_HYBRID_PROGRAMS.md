# E023 — Programas híbridos

Protocolo pré-registrado em `EXPERIMENTS.md`. Estado inicial: commit
`5be5ecb908bb0a8bbdd89be62ee08cdb2e63d01b`, árvore suja herdada preservada.

Os programas adicionais são ortogonais aos 32 programas gerais, sem girar
essa base original. Os candidatos incluem deltas por grupo, deltas de
pseudobulk ponderado por abundância e partes positivas/negativas dos deltas
ponderadas pela estabilidade de sinal entre transições. Somente as seis
transições de treino com alvos E7.0–E8.25 entram na construção. E8.5 externo
continua usado para seleção da época, como no E021: não é teste independente.
O cache de agrupamento herdado também inclui esse holdout.

## Comandos reproduzíveis

Executados no diretório raiz. Cada diretório deve ser novo para preservar runs.

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e021_anchored --experiment E023 --output-dir outputs/population_transformer/E023/general32_de16_seed42_n512 --n-programs 48 --de-programs 16 --n-cells 512 --epochs 250 --patience 25 --seed 42 --program-seed 42 --evaluation-seed 42 --device cuda --scales 0.25 --proportion-scale 0.25
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e021_anchored --experiment E023 --output-dir outputs/population_transformer/E023/general32_de32_seed42_n512 --n-programs 64 --de-programs 32 --n-cells 512 --epochs 250 --patience 25 --seed 42 --program-seed 42 --evaluation-seed 42 --device cuda --scales 0.25 --proportion-scale 0.25
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e021_anchored --experiment E023 --output-dir outputs/population_transformer/E023/general64_seed42_n512 --n-programs 64 --de-programs 0 --n-cells 512 --epochs 250 --patience 25 --seed 42 --program-seed 42 --evaluation-seed 42 --device cuda --scales 0.25 --proportion-scale 0.25
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e021_anchored --experiment E023 --output-dir outputs/population_transformer/E023/general32_seed42_n512 --n-programs 32 --de-programs 0 --n-cells 512 --epochs 250 --patience 25 --seed 42 --program-seed 42 --evaluation-seed 42 --device cuda --scales 0.25 --proportion-scale 0.25

for config in general32_de16_seed42_n512 general32_de32_seed42_n512 general32_seed42_n512; do
  UV_CACHE_DIR=/tmp/ve-uv-cache uv run veckit --task T1 --input outputs/population_transformer/E023/$config/anchored_0.25.h5ad --target outputs/population_transformer/E021/fixed_eval_seed42/target.h5ad --reference outputs/population_transformer/E021/fixed_eval_seed42/reference.h5ad --seed 42 --out outputs/population_transformer/E023/$config/veckit.json
done
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.audit_e023
UV_CACHE_DIR=/tmp/ve-uv-cache uv run pytest tests/test_e023_programs.py tests/test_population_transformer_v2.py tests/test_population_pipeline_v2.py tests/test_population_temporal.py -q
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m compileall -q src/approaches/population_transformer src/scripts/run_e021_anchored.py src/scripts/audit_e023.py tests/test_e023_programs.py
git diff --check
```

## Tentativas inválidas e assistência mecânica

64 gerais falhou antes do treino: existem somente 37 linhas de mudanças
observadas. Não foi completado com direções arbitrárias nem utilizado o alvo.
Não houve OOM nem necessidade de servidor.

Ollama `qwen3-coder:30b` foi consultado em
`http://localhost:11434/v1/chat/completions` somente para sugerir testes
numéricos pequenos, sem dados, código privado, checkpoints ou segredos.
Resposta descartada: propunha mais vetores ortogonais do que a dimensão
permitia e uma comparação de matrizes com shapes incompatíveis. Nenhum código
retornado foi incorporado. Os testes locais verificam preservação exata da
base geral, ortogonalidade, determinismo e menor erro de projeção sintético.

## Artefatos

Cada run válido contém população `.h5ad`, alvo/referência, `checkpoint.pt`,
`programs.npz`, `config.json`, `history.json`, `metrics.json` e `veckit.json`.
`outputs/population_transformer/E023/audit.json` registra comparação exata
de alvo/referência com E021, contrato, genes, unicidade, zeros, ortogonalidade,
SHA-256 e tamanhos. Nenhum arquivo E10.5 real foi lido.

## Resultado e encerramento

O retreino de 32 programas reproduziu exatamente a matriz de predição E021
(igualdade elemento a elemento) e as métricas veckit. As bases 32+16 e 32+32
obtiveram DE 0,0286 e 0,0095 contra 0,0476 do E021; ambas também pioraram
direção, MMD e variograma. Tabela completa em `EXPERIMENTS.md`.

Condição de parada atingida: nenhuma configuração nova passou os gates.
Não houve teste de 1.024 células, seeds 43/44 ou ensemble, nem geração de
E10.5. A estabilidade das híbridas permanece não medida. E021 é preservado,
com sua limitação conhecida de estabilidade parcial. Recomenda-se um
diagnóstico de capacidade versus previsibilidade em transições internas
antes de mudar o encoder. Nenhuma submissão externa realizada.

Arquivos alterados neste ciclo: `anchored.py`, `run_e021_anchored.py`,
`audit_e023.py`, `test_e023_programs.py`, `EXPERIMENTS.md`, `STATUS.md`
e este relatório, além dos artefatos novos em E023. Alterações preexistentes
em outros arquivos foram preservadas. Testes: 18 passaram; compileall,
contratos e `git diff --check` passaram.
