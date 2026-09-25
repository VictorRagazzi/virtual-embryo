# M2 — Embeddings, agrupamento e tokens populacionais

**Data:** 2026-09-24  
**Situação:** concluído em caráter exploratório; não submetível.

## Auditoria dos dados adicionais

| Arquivo | Células | Estágio | Theiler | Tipos celulares | Faixa X | Densidade |
|---|---:|---|---|---:|---:|---:|
| `E65_ex.h5ad` | 4 | E6.5 | TS9 | 1 | 0–5,3240 | 7,09% |
| `E675_ex.h5ad` | 158 | E6.75 | TS9 | 2 | 0–4,9391 | 6,60% |
| `E70_ex.h5ad` | 1.897 | E7.0 | TS10 | 2 | 0–5,5018 | 9,12% |
| `E725_ex.h5ad` | 2.449 | E7.25 | TS10 | 2 | 0–5,3498 | 8,81% |
| `E75_ex.h5ad` | 1.336 | E7.5 | TS11 | 3 | 0–5,2830 | 9,09% |
| `E775_ex.h5ad` | 811 | E7.75 | TS11 | 4 | 0–5,7972 | 9,42% |
| `E80_ex.h5ad` | 971 | E8.0 | TS12 | 3 | 0–5,7964 | 9,02% |
| `E825_ex.h5ad` | 1.308 | E8.25 | TS12 | 2 | 0–6,0837 | 9,33% |
| `E85_ex.h5ad` | 1.306 | E8.5 | TS12 | 2 | 0–7,2296 | 9,22% |

Todos são densos em disco, `float32`, finitos e não negativos, com 32.285 genes exatamente na ordem oficial. `uns['log1p']` existe e `sum(expm1(X))=10.000` por célula na amostra auditada, portanto a escala é `log1p(CP10k)`. Os metadados incluem amostra, estágio, Theiler, batch, clusters, doublet e tipo celular. Todos têm `doublet=False`. Não foram encontrados IDs repetidos entre `_ex`, D001 ou D002. A origem e os hashes estão registrados em `DATA_POLICY.md`.

## Contrato e execução

O cache contém 44.084 CLS `float32 [n,512]` do M001 congelado. O agrupamento é conjunto entre dez estágios. Para cada `(estágio, grupo)`, o artefato guarda contagem, proporção, média e desvio-padrão latentes, 256 pseudobulks gênicos, dez top genes, máscara de vazio e resíduos célula menos média do grupo no estágio. Os escores gênicos são um baseline simples, não módulos biológicos aprendidos.

K=50 foi escolhido porque apresentou a maior estabilidade da grade (ARI médio 0,4248) e menor variação de proporções casadas, embora a estabilidade absoluta seja somente moderada. K=100/200 diminuíram a inércia, mas fragmentaram mais os estágios. A tabela completa e os comandos estão em E014.

## Validação e limitações

- 44.084/44.084 células possuem rótulo em cada K;
- todos os K grupos globais aparecem;
- proporções por estágio somam 1 com erro máximo `1,19e-7`;
- centros, dispersões, programas e resíduos são finitos;
- `13 passed` nos testes M0/M1/M2 selecionados;
- E6.5 possui somente quatro células e não sustenta estimativas populacionais robustas;
- os `_ex` cobrem apenas quatro rótulos mesodérmicos/cardiogênicos e não representam o atlas integral;
- M001 continua exploratório/não submetível e o decoder cru continua com colapso de diversidade;
- nenhum Transformer temporal foi implementado ou treinado.

Próximo passo recomendado: aguardar autorização para M3 e, antes de qualquer dinâmica temporal, testar se K=50 reconstrói uma população conhecida por centros + resíduos sem agravar o colapso do decoder.
