# Mouse-Geneformer temporal

Pipeline independente em `src/approaches/llm/T2`, adaptado do código local de
`../mouse_geneformer`. Recebe expressão, estágio `t` e intervalo `Δt` em dias.

```text
expressão → tokens por rank/mediana → Mouse-Geneformer → média com máscara
                                                        + t + Δt
                                                            ↓
                                                   regressor de expressão
```

Os embeddings e os primeiros blocos ficam congelados; os dois últimos blocos
são ajustados por padrão. `--mode delta` prevê a mudança em expressão log
normalizada e soma à origem. `--mode direct` prevê a expressão futura diretamente.
A saída inclui todos os genes comuns aos estágios, inclusive genes sem token.
As predições exportadas são truncadas em zero.

## Dados e separação

`experiment prepare` descobre os nomes explícitos E65, E675, E70, E725, E75,
E775, E80, E825 com sufixo `_ex.h5ad`, e os oficiais E85/E95.
Na árvore atual há nove estágios, **E6.75–E9.5**; E6.5 não está disponível.
Em E8.5, usamos o arquivo oficial; `E85_ex.h5ad` fica fora para manter uma
única fonte por estágio e evitar possível duplicação.

1. Reserva aleatoriamente 2.000 IDs de E8.5 e outros 2.000 de E9.5, com seed fixa.
2. Exclui esses IDs de todos os estágios antes de amostrar, normalizar, fazer SVD
   ou construir pares. A reserva não participa da escolha do checkpoint.
3. Divide as células restantes em treino/validação interna (80/20).
4. Ajusta SVD somente no treino e cria pseudo-pares dentro de cada split.
5. Treina todas as combinações crescentes: nove estágios geram 36 transições.
   `TemporalPairs` fornece `[t, destino - origem]` para cada par.
6. Escolhe o menor MSE de validação interna e avalia o checkpoint uma vez sobre
   as duas amostras reservadas.

`data.py::pair_cells(source_expression, future_expression,
strategy="nearest_neighbor")` é o ponto de troca do pareamento. Recebe duas
matrizes na mesma projeção SVD e retorna um índice de destino e uma distância
por origem. O padrão usa similaridade de expressão, permite destinos repetidos
e não restringe tipos celulares. Estratégias desconhecidas geram erro.

## Preparar o ambiente

Execute todos os comandos deste documento na raiz do repositório. O projeto usa
Python 3.11 e o `uv` cria e sincroniza o ambiente virtual a partir do
`pyproject.toml` e do `uv.lock`:

```bash
uv sync
```

## Recursos oficiais

Baixe os recursos do Mouse-Geneformer em `models/mouse-Geneformer`:

```bash
uv run python script/download_model.py \
  --output-dir models/mouse-Geneformer
```

O script baixa os pesos e os três dicionários, gera `gene_map.csv` e registra
`resources.json`. Trata a confirmação do Google Drive e verifica SHA-256 antes
de disponibilizar cada arquivo. Uma nova execução reutiliza arquivos válidos e
baixa apenas os ausentes; arquivos existentes divergentes geram erro e são
preservados. É necessário acesso à internet ao Google Drive e ao Hugging Face.

Pesos: [Mouse-Geneformer base, seis blocos](https://github.com/machine-perception-robotics-group/Mouse-Geneformer#trained-model).
Dicionários: [Mouse-Genecorpus-20M dos autores](https://huggingface.co/datasets/MPRG/Mouse-Genecorpus-20M/tree/main),
conforme a [resposta sobre arquivos ausentes](https://github.com/machine-perception-robotics-group/Mouse-Geneformer/issues/1).

Os recursos baixados ficam em `models/mouse-Geneformer/`:

- `config.json` e `pytorch_model.bin`;
- `MLM-re_token_dictionary_v1.pkl`;
- `mouse_gene_median_dictionary.pkl`;
- `MLM-re_token_dictionary_v1_GeneSymbol_to_EnsemblID.pkl`;
- `gene_map.csv`, conversão determinística do último dicionário;
- `resources.json`, URLs e hashes SHA-256.

Os pickles são recursos dos autores. O treino exige pesos completos compatíveis,
e nunca substitui pesos ausentes por inicialização aleatória. Tokenização usa
somente genes expressos, ordenados por expressão linear/mediana oficial, sem
CLS/SEP, com máscara de padding. `--max-length` aceita até 2.048 genes.

## Escala de expressão

Declare `--expression-scale counts` para contagens inteiras ou `log1p` para log
natural de expressão linear. Não use matrizes centradas, z-score ou corrigidas
por lote. `--layer` permite selecionar uma layer comum aos arquivos.

Os arquivos `_ex` foram normalizados para 10.000 e transformados com log1p pelo
script local de extração. E85/E95 não registram essa transformação. Em uma
inspeção de 32 células por arquivo, as somas de `expm1(X)` ficaram entre
9999,749 e 10000,445, compatíveis com log1p(CP10k). **Isso é uma inferência
numérica, não confirmação da preparação original.** O piloto usa `log1p`.

Os alvos, a referência E8.5 e o alvo E9.5 da avaliação usam a mesma escala:
`log1p` após normalização para soma 10.000 antes de selecionar genes comuns.

## Executar da raiz do projeto

Preparar a reserva (recusa sobrescrever um manifesto existente):

```bash
uv run python -m src.approaches.llm.T2.experiment prepare \
  --data-dir data --output data/T2/manifest.json
```

Treino piloto em CPU, com os pesos reais:

```bash
uv run python -m src.approaches.llm.T2.temporal train \
  --manifest data/T2/manifest.json \
  --encoder models/mouse-Geneformer \
  --tokens models/mouse-Geneformer/MLM-re_token_dictionary_v1.pkl \
  --medians models/mouse-Geneformer/mouse_gene_median_dictionary.pkl \
  --gene-map models/mouse-Geneformer/gene_map.csv \
  --expression-scale log1p --mode delta --trainable-layers 2 \
  --epochs 3 --batch-size 8 --max-cells 128 --max-length 512 \
  --components 30 --device cpu --output models/T2_temporal
```

`--max-cells` limita as células por estágio após excluir a reserva (padrão
2.000); não altera seu tamanho. A execução piloto reduz esse valor para 128.
Para um treino maior, aumente células/épocas e use até 2.048 tokens, conforme os
recursos disponíveis. `--device cuda` exige uma GPU; o ambiente atual só tem CPU.
Se o treino em CPU deixar a máquina pesada, prefixe o comando com
`OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4` para limitar threads numéricas.
`--group-column` permite manter embriões/réplicas no mesmo split interno quando
a coluna existe em todos os estágios. Isso não transforma a reserva aleatória
por células em uma avaliação independente por embrião.

Saídas: `best.pt`, `history.json`, `pairs.csv` e `settings.json`. O checkpoint
inclui os genes, recursos de tokenização e o manifesto exato da reserva.

Avaliação solicitada, sempre limitada a no máximo 2.000 células por arquivo:

```bash
uv run python -m src.approaches.llm.T2.experiment evaluate \
  --checkpoint models/T2_temporal/best.pt \
  --output data/T2/evaluation --batch-size 8 --device cpu
```

Produz `prediction.h5ad`, `target.h5ad`, `reference.h5ad`, `source_raw.h5ad`,
`manifest.json` e `metrics.json`. Usa `veckit.score(task="T1", input=prediction,
target=target, reference=reference)`: **T1 é a task temporal do veckit**, enquanto
T2 é o nome desta pasta. Não fornece rótulos futuros inventados ao avaliador.
A biblioteca pode inferi-los a partir da expressão predita.

Os arquivos originais são abertos em modo `backed`; somente as linhas amostradas
são materializadas. A avaliação recusa amostras acima de 2.000 e não passa os
arquivos completos ao veckit. Para outra execução, use um diretório de saída
novo. Reutilizar resultados para ajustar hiperparâmetros consome a independência
da reserva como teste final.

Inferência avulsa (a entrada deste comando deve estar previamente amostrada se
for usada para avaliação):

```bash
uv run python -m src.approaches.llm.T2.temporal predict \
  --checkpoint models/T2_temporal/best.pt \
  --input data/T2/evaluation/source_raw.h5ad --time 8.5 --delta-time 1.0 \
  --output data/T2/prediction_again.h5ad
```

## Testes e limites

```bash
uv run pytest -q tests/test_T2.py
```

Testes cobrem rank/medianas, congelamento, máscara, direto/delta, dependência de
`t` e `Δt`, intervalos irregulares, separação antes dos pares, reserva reproduzível,
limite de memória por amostra e treino/checkpoint/inferência com BERT minúsculo
aleatório **somente nos testes**.

Vizinhos são pseudo-pares, não descendentes observados. O modelo produz uma
previsão determinística por origem; não modela explicitamente destinos múltiplos,
proliferação ou morte. A avaliação por células mede distribuição E8.5→E9.5 em
estágios presentes no treino; não demonstra generalização para outro embrião ou
E10.5. O treino piloto é uma execução inicial, não um ajuste exaustivo.

## Execução realizada

Veja [resultados do piloto e auditoria](RESULTS.md), incluindo a comparação real
com veckit sobre 2.000 células por estágio.
