# Resultado do piloto — 15/09/2026

Treino real com Mouse-Geneformer pré-treinado, em CPU. Nove estágios de E6.75 a
E9.5, 36 transições, 128 células por estágio (1.152 no total), 512 tokens,
três épocas, modo delta e ajuste dos dois últimos blocos. O pareamento usa
vizinhança em SVD de 30 componentes, ajustada somente no treino.

**A comparação de distribuições foi executada com 2.000 predições e 2.000 células
reais de E9.5 reservadas antes do treino.** A referência é a amostra de 2.000
células E8.5 usada como origem da previsão. Nenhum ID da reserva aparece nos
pares de treino ou de validação interna. Todos os arquivos avaliados têm os
mesmos 32.285 genes, na mesma ordem; as predições são finitas e não negativas.

## Validação interna nos pseudo-pares

| Época | MSE de treino | MSE de validação | MSE de persistência |
| --- | --- | --- | --- |
| 1 | 0.075797 | 0.074857 | 0.080834 |
| 2 | 0.073784 | 0.074471 | 0.080834 |
| 3 | 0.073438 | 0.074263 | 0.080834 |

Melhor checkpoint: época 3. Redução relativa de MSE de 8.13% sobre copiar a expressão de origem.

## Comparação independente com veckit 0.1.2

`score(task="T1", input=prediction, target=target, reference=reference, seed=42)`.
Não foi calculada pontuação agregada de leaderboard.

| Métrica | Valor |
| --- | ---: |
| `de_score` | 0.0167 |
| `de_direction` | 0.0174 |
| `energy_distance` | 0.93261 |
| `mmd_u` | 0.02964 |
| `variogram` | 0.009592 |
| `pb_rel_err` | 0.1357 |
| `library_size_ratio` | 1.071 |
| `variance_ratio` | 0.931 |
| `composition_JSD` | 0.0501 |
| `pseudobulk_pearson` | 0.9907 |

**Interpretação:** a correlação de expressão média é alta, mas a recuperação de
genes diferencialmente expressos e a direção da mudança são baixas. O piloto
não demonstra boa previsão da mudança temporal. A melhora de MSE nos
pseudo-pares não substitui essa avaliação de distribuições.

## Artefatos locais

- [Checkpoint](../../../../models/T2_temporal/best.pt).
- [Histórico](../../../../models/T2_temporal/history.json) e [configuração](../../../../models/T2_temporal/settings.json).
- [Pares](../../../../models/T2_temporal/pairs.csv).
- [Métricas completas](../../../../data/T2/evaluation/metrics.json).
- [Auditoria](../../../../data/T2/evaluation/audit.json) e [IDs reservados](../../../../data/T2/evaluation/manifest.json).
- [Predição](../../../../data/T2/evaluation/prediction.h5ad), [alvo](../../../../data/T2/evaluation/target.h5ad) e [referência](../../../../data/T2/evaluation/reference.h5ad).
- [Recursos e hashes](../../../../models/mouse-Geneformer/resources.json) e [versões](../../../../models/mouse-Geneformer/environment.json).

Os artefatos de dados e modelos ficam em pastas ignoradas pelo Git. Este relatório
preserva o resumo; os comandos de reprodução estão no [README](README.md).

## Limitações

- Execução piloto: 128 células por estágio, três épocas e truncamento em 512 tokens.
- E6.5 ausente; cobertura efetiva E6.75–E9.5.
- Escala oficial inferida numericamente como log1p(CP10k), sem confirmação da origem.
- Vizinhos não são descendentes observados; divisão por células não testa generalização por embrião.
- E8.5/E9.5 aparecem no treino com outras células; não se mediu extrapolação a um estágio ausente.
- A reserva já foi avaliada. Usar suas métricas para escolher novas configurações exige outro teste independente.

## Verificações

Nove testes em `tests/test_T2.py` passaram. A auditoria verificou dimensões,
IDs, tempos, ordem dos genes, valores finitos/não negativos e ausência de
sobreposição entre reserva e pares. `git diff --check` não encontrou problemas.
