# E028 — Treino somente oficial e capacidade do Transformer

E028 removeu completamente os arquivos `_ex` e usou apenas E8.5/E9.5 oficiais. Como isso deixa uma única transição, o experimento é um diagnóstico de capacidade em células reservadas do mesmo par temporal, não uma avaliação de extrapolação para E10.5.

Cada estágio foi separado em 1.024 células de treino, 256 de validação e 512 de teste. Sob a mesma PCA não linear, decoder com resíduo e loss populacional, foram comparados Transformers de 71 mil, 538 mil e 3,18 milhões de parâmetros com um deslocamento médio latente.

| Modelo | DE | direção | MMD | variograma |
|---|---:|---:|---:|---:|
| deslocamento médio | 0,6200 | 0,4759 | 0,03006 | 0,002763 |
| pequeno | 0,4400 | 0,3182 | 0,04106 | 0,004546 |
| médio | 0,4200 | 0,2806 | 0,05036 | 0,006458 |
| grande | 0,3200 | 0,2246 | 0,11061 | 0,016895 |

A complexidade adicional piorou monotonicamente validação e teste. Todos os arquivos passaram o contrato e preservaram 512 células únicas. Artefatos: `outputs/population_transformer/E028/seed42_n512/`.

Conclusão: não aumentar o Transformer neste regime. Remover `_ex` impede medir generalização temporal e, por isso, não há evidência suficiente para substituir o treino atual da candidata E10.5.
