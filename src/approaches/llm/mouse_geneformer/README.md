# Mouse-Geneformer temporal

Implementação independente de `llm/T1`: expressão em `t` → encoder Mouse-Geneformer → média dos embeddings dos genes expressos → regressor condicionado por `t` e `Δt` → expressão futura.

## Situação dos dados locais

- `data/E85.h5ad`: 16.787 células, 32.285 genes por símbolo.
- `data/E95.h5ad`: 17.057 células, os mesmos 32.285 genes.
- Ambos têm `obs.celltype`, mas não têm contagens em layers, IDs Ensembl ou registro da transformação de X. X contém valores fracionários: **não declarar `counts` e não assumir `log1p` sem confirmar a preparação original**.
- `data/huggingface/cell_index.parquet` é um índice: os 135 caminhos de expressão que referencia não estão no pacote. Inclui referências a estágios embrionários, mas não permite reconstruir suas matrizes.
- Os dois h5ad baixados são experimentos de hematopoiese (Weinreb) e diferenciação pancreática (Veres). Seus dias experimentais **não são estágios embrionários**. O uso de nomes de genes ortólogos de camundongo também não comprova a espécie das células originais.

A separação foi executada, preservando expressão, ordem dos genes e metadados. `celltype` e `timepoint` foram acrescentados. Cada pasta tem um `manifest.json` com origem, dia e dimensões.

| Pasta | Dia: número de células | Genes |
| --- | --- | --- |
| `data/by_day/weinreb/` | 2: 28.249; 4: 48.498; 6: 54.140 | 14.978 |
| `data/by_day/veres/` | 0: 6.190; 1: 6.177; 2: 3.562; 3: 12.477; 4: 6.480; 5: 6.404; 6: 6.013; 7: 3.971 | 13.660 |

Os nomes são `day_2.h5ad`, por exemplo. As saídas ficam sob `data/`, que já é ignorada pelo Git. Para reproduzir, escolha diretórios novos; o script recusa sobrescrita:

```bash
.venv/bin/python src/scripts/split_by_day.py \
  data/huggingface/weinreb_ortholog.h5ad data/by_day/weinreb \
  --time-column 'Time point' --celltype-column cell_type

.venv/bin/python src/scripts/split_by_day.py \
  data/huggingface/veres_ortholog.h5ad data/by_day/veres \
  --time-column 'Time point' --celltype-column cell_type
```

## Preparar o modelo

Instale a dependência adicional sobre o ambiente do projeto:

```bash
uv pip install --python .venv/bin/python -r src/approaches/mouse_geneformer/requirements.txt
```

Validado com `transformers==4.57.6`. As dependências e o lockfile existentes foram preservados.

Obtenha o checkpoint e os recursos compatíveis na [implementação oficial do Mouse-Geneformer](https://github.com/machine-perception-robotics-group/Mouse-Geneformer). O README oficial aponta o [arquivo do modelo base](https://drive.google.com/file/d/1gM3gcc3DlNGt5bAcqHbeRxtdMktGeDEg/view). São necessários:

1. Diretório local Hugging Face BERT com `config.json` e pesos pré-treinados. Não há fallback para inicialização aleatória no comando de treino.
2. Dicionário gene → token, incluindo `<pad>`.
3. Dicionário gene → mediana do corpus de pré-treino correspondente. Não substituir por medianas do experimento.
4. IDs compatíveis: coluna `var.ensembl_id`, nomes de genes já compatíveis ou CSV `gene_symbol,ensembl_id` passado em `--gene-map`. Para E85/E95, falta esse mapeamento.

Dicionários podem ser JSON ou pickle de fonte confiável. O tokenizer ordena apenas genes expressos por expressão linear/mediana e limita a 2.048 tokens, sem CLS/SEP, seguindo o [tokenizer oficial](https://github.com/machine-perception-robotics-group/Mouse-Geneformer/blob/master/geneformer/tokenizer.py). Padding fica fora da média dos embeddings. Genes sem correspondência ficam fora do encoder; a saída inclui todos os genes comuns aos estágios, inclusive aqueles sem token.

`--expression-scale counts` exige contagens inteiras não negativas. `--expression-scale log1p` aplica `expm1` e pressupõe log natural sobre expressão linear, sem scaling, correção de lote ou z-score. Para tokens, o fator multiplicativo por célula não altera a ordenação. Os alvos são sempre `log1p` da expressão normalizada para soma 10.000 **antes** de selecionar genes comuns. A saída não é contagem bruta. `--layer counts` pode selecionar contagens presentes em uma layer; o mesmo nome deve existir na inferência.

## Treinar

Execute da raiz do projeto. Os caminhos de recursos abaixo são exemplos a preencher. O exemplo com `log1p` só se aplica após confirmar essa escala nos arquivos:

```bash
.venv/bin/python -m src.approaches.mouse_geneformer.temporal train \
  --stage 8.5=data/E85.h5ad --stage 9.5=data/E95.h5ad \
  --encoder models/mouse-Geneformer \
  --tokens models/mouse-Geneformer/token_dictionary.pkl \
  --medians models/mouse-Geneformer/gene_median_dictionary.pkl \
  --gene-map data/mouse_gene_map.csv \
  --expression-scale log1p --mode delta --trainable-layers 2 \
  --epochs 10 --batch-size 8 --device cpu \
  --output models/mouse_geneformer_temporal
```

Adicione `--stage 6.5=... --stage 7.5=...` quando tiver matrizes embrionárias compatíveis. O treino usa todas as combinações de tempos crescentes, incluindo intervalos não adjacentes. Não junte experimentos distintos só porque os tempos têm valores parecidos.

- `--mode delta` (padrão): aprende mudança na escala log normalizada e soma à expressão inicial.
- `--mode direct`: prevê expressão futura na mesma escala.
- `--trainable-layers 0`: congela todo o encoder; 2 ajusta somente os dois últimos blocos. Embeddings ficam congelados.
- `--max-cells 200`: amostra por estágio com seed fixa para testar o fluxo com menos custo. Omita para usar todas as células.
- `--device cuda`: utiliza GPU, se disponível; sequências de 2.048 genes podem exigir reduzir batch size.
- `--group-column embryo_id`: separa embriões/réplicas inteiros e mantém o grupo no mesmo split entre estágios. Requer metadados reais; E85/E95 atuais não os contêm.

O diretório de saída deve estar vazio. Contém `best.pt`, `history.json`, `settings.json` e `pairs.csv` com origem, destino, tempos, split e distância. O checkpoint guarda os recursos de tokenização, genes de saída e transformação para reutilizar na inferência.

## Pareamento e avaliação

`data.py::select_future_neighbors(source_expression, future_expression)` é o ponto de troca solicitado: retorna um índice de destino e uma distância para cada origem. Hoje escolhe o vizinho euclidiano mais próximo e permite que várias origens tenham o mesmo destino. Não restringe por tipo celular.

O fluxo separa células de treino/validação **antes** de construir pares. Uma SVD truncada é ajustada somente nas células de treino, usando expressão log normalizada dos genes comuns; `--components` controla a dimensão (padrão 50). A validação usa essa mesma projeção e busca vizinhos apenas entre destinos de validação. Não é uma PCA centrada.

As métricas comparam a predição com os pseudo-alvos e com o baseline de persistência (`X_futuro = X_atual`). O melhor checkpoint minimiza o MSE de validação. Valores negativos são truncados para zero na avaliação e na exportação; o treino mantém a regressão sem truncamento.

**Limites metodológicos:** vizinhança não demonstra ancestralidade; pode favorecer pouca mudança e confundir lotes. A validação por células não mede generalização para embriões independentes ou tempos ausentes. Os rótulos observados não são tipos futuros previstos. Com somente E8.5/E9.5, não há evidência de generalização E6.5–E9.5 ou E10.5. Para medir isso, será necessário um experimento com estágios/embriões externos ao treino e avaliação biológica além do erro nos pseudo-pares.

## Prever

```bash
.venv/bin/python -m src.approaches.mouse_geneformer.temporal predict \
  --checkpoint models/mouse_geneformer_temporal/best.pt \
  --input data/E85.h5ad --time 8.5 --delta-time 1.0 \
  --output predictions/mouse_geneformer_E95.h5ad
```

A entrada precisa conter todos os genes de saída do treino. O h5ad exportado contém somente esses genes, preserva IDs das células de origem e registra tempo inicial, futuro e escala. `celltype`/`cell_type` tornam-se `source_celltype`/`source_cell_type`. A inferência aceita intervalos positivos, mas intervalos/estágios não vistos são extrapolação sem validação automática.

## Validação executável

```bash
.venv/bin/python -m pytest -q tests/test_mouse_geneformer.py
```

Testa separação por dia, preservação da expressão, vizinhança, rank por medianas, normalização, separação antes dos pares, congelamento, máscara de padding, modos direto/delta e um ciclo treino → checkpoint → predição com um BERT pequeno aleatório **somente no teste**. O treino científico com os pesos oficiais ainda depende dos recursos e da escala de entrada descritos acima.
