# E029 — Mudança E8.5→E9.5 por grupos compartilhados

Um K-means conjunto com K=32 foi ajustado somente nas células de treino oficiais E8.5 e E9.5. O grupo `k` em ambos os estágios é a mesma região do espaço latente. Cada token contém proporção, centro e dispersão; o Transformer aprende `estado_k(E8.5) → estado_k(E9.5)`. Não há pareamento de células nem alegação de linhagem.

O teste usa células reservadas: 2.048 E8.5 formam o pool de âncoras para 512 saídas e outras 512 E9.5 formam o alvo. A reconstrução usa o decoder com resíduo empírico.

| Modelo | DE | direção | MMD | variograma |
|---|---:|---:|---:|---:|
| copiar grupos | 0,0645 | 0,2874 | 0,03487 | 0,001206 |
| delta por grupo | 0,6129 | 0,5140 | 0,02336 | 0,002486 |
| Transformer pequeno | 0,4516 | 0,4778 | 0,01676 | 0,004058 |
| Transformer médio | 0,3387 | 0,4197 | 0,02230 | 0,004916 |
| Transformer grande | 0,1613 | 0,2323 | 0,02996 | 0,005172 |

Os grupos capturam mudança útil na transição conhecida. O modelo pequeno é o melhor Transformer; aumentar a complexidade piora. O delta direto é o melhor método na maioria das métricas e deve permanecer baseline obrigatório. Como só há uma transição, o experimento não mede extrapolação E10.5.

Artefatos: `outputs/population_transformer/E029/seed42_k32_n512/`.

## Candidatas E10.5

Por solicitação do responsável, E029 foi extrapolado de E9.5 para E10.5 com 2.500 células. Nenhum dado E10.5 foi lido. `group_delta` reaplica diretamente o delta E8.5→E9.5 por grupo; `small` usa o Transformer pequeno da época 246. As células E9.5 são âncoras únicas selecionadas de um pool de 5.000 e a saída é normalizada para `log1p(CP10k)`.

- `final_seed42/e10_5_group_delta_2500.h5ad`: 2.500 células únicas; 24 grupos; zeros 60,73%; máximo 6,4145.
- `final_seed42/e10_5_small_2500.h5ad`: 2.500 células únicas; 32 grupos; zeros 59,89%; máximo 9,0876.

Ambos passam o contrato estrutural. O máximo e a densidade da variante `small` evidenciam extrapolação mais agressiva; não existe alvo local permitido para determinar qual terá melhor score oficial.
