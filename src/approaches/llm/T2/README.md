# Mouse-Geneformer temporal

Pipeline em `src/approaches/llm/T2` para o Mouse-Geneformer temporal. Recebe
expressão, estágio `t` e intervalo `Δt` em dias.

```text
expressão → tokens por rank/mediana → Mouse-Geneformer → média com máscara
                                                        + t + Δt
                                                        + expressão projetada (opcional)
                                                            ↓
                                                 saída por gene
                                                            ↓
                              delta: origem + α × saída
                              direct: origem + α × (saída − origem)
```

Os embeddings e os primeiros blocos ficam congelados; os dois últimos blocos
são ajustados por padrão. `--mode delta` (padrão) aprende uma mudança aditiva;
`--mode direct` aprende a expressão futura. `α=0` retorna a entrada na escala
interna `log1p(CP10k)`; `Δt` é uma entrada da cabeça, mas zero não obriga a
predição a ser idêntica à origem. `α` interpola ou escala a saída sem retreino.
O clamp em zero na exportação preserva a identidade com `α=0`, pois a entrada
interna é não negativa; isso não implica igualdade com as contagens brutas.
Checkpoints de velocidade da arquitetura 2 continuam legíveis, com sua fórmula
original `origem + α × Δt × velocidade`. Checkpoints sem versão são rejeitados.
Os números e artefatos históricos em [RESULTS.md](RESULTS.md) descrevem a
arquitetura anterior e não são resultados desta revisão.
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
6. Escolhe a menor perda de validação interna e avalia o checkpoint uma vez sobre
   as duas amostras reservadas.

`--pairing nearest_neighbor` é o padrão: escolhe o destino mais próximo na SVD,
permite destinos repetidos e não restringe tipos celulares. `--pairing ot` calcula
um acoplamento entrópico por transição e split, com marginais empíricas uniformes,
e amostra um destino por origem da sua linha normalizada. A mesma SVD é ajustada
somente no treino; a validação usa a projeção treinada, mas calcula seu próprio
acoplamento sem células de treino. `--ot-regularization` (padrão 0,1) multiplica
a mediana dos custos quadráticos de cada transição. Ajuste esse valor somente
na validação interna. OT não usa igualdade de tipo celular como restrição.

## Entrada quantitativa e perda

`--expression-input none` reproduz a entrada da cabeça anterior: resumo do
encoder, `t` e `Δt`. `projected` acrescenta uma projeção aprendida de todos os
genes alinhados para 16 números por célula. Ela recebe magnitudes reais em
`log1p(CP10k)`, inclusive genes sem token; acrescenta `16 × número de genes`
pesos, sem nova camada profunda. A ordem dos genes é salva no checkpoint e
reaplicada na predição. `none` é o padrão para comparar a mudança isolada.

A perda é `MSE(pseudo-par) + --mmd-weight × MMD(população)`, com peso zero por
padrão. O MSE usa o pseudo-par escolhido para cada origem. Para MMD, cada
lote contém apenas uma transição `(t, t+Δt)` e compara suas predições com uma
amostra **uniforme das células reais do destino no mesmo split**, antes do
pareamento. Assim, destinos repetidos nos pseudo-pares não alteram a frequência
da população real. O controle com peso zero usa o mesmo agrupamento dos lotes,
para isolar o efeito da perda. Os lotes finais de uma célula são unidos ao anterior.
Com peso positivo, são necessárias ao menos duas origens por transição em cada
split e `batch-size ≥ 2`.
A MMD usa kernel RBF na expressão completa, banda mediana das distâncias entre
as células reais do lote e estimador enviesado, estável para lotes pequenos.
`train_mmd` e `validation_mmd` são registrados também com peso zero quando há
ao menos duas células por lote e por população de destino; ficam nulos se não
houver lote elegível. Assim, o controle sem MMD pode ser comparado na validação.
Não é numericamente a métrica `mmd_u` do avaliador. O peso multiplica a MMD
sem normalização automática; compare MSE, MMD e métricas de avaliação ao
escolhê-lo **somente com dados de treino/validação interna**. A seleção do
checkpoint usa essa mesma perda combinada na validação interna.

No `veckit` instalado localmente, `de_score` compara genes de mudança com
direção correta contra um nulo baseado na expressão de referência;
`de_direction` mede correlação parcial dos ranks das mudanças, controlando a
expressão de referência; `mmd_u` é MMD não enviesada com múltiplas bandas em
PCA ajustada no alvo observado; `variogram` compara médias de diferenças de
expressão entre pares de genes. Esses termos avaliam propriedades que o MSE
dos pseudo-pares não mede diretamente. A alternativa acima adiciona somente
um termo simples de distribuição, sem reproduzir o scorer nem ajustar PCA,
bandas ou genes com células reservadas.

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

Preparar a reserva (o manifesto no caminho indicado é sobrescrito):

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
  --expression-scale log1p --trainable-layers 2 --mode delta \
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

Saídas: `best.pt`, `history.json`, `pairs.csv` e `settings.json`. Com OT, também
há `ot/summary.json` e uma matriz `.npz` por transição e split. Cada matriz
inclui pesos, massas de linhas/colunas, entropia por origem e IDs na ordem da
matriz. O resumo registra tempo, memória da matriz, distâncias, reutilização de
destinos, regularização, intervalo temporal e proporções celulares quando há
rótulos. Uma nova
execução no mesmo diretório sobrescreve esses arquivos e preserva outros. O
checkpoint inclui os genes, recursos de tokenização e o manifesto da reserva.

Avaliação solicitada, sempre limitada a no máximo 2.000 células por arquivo:

```bash
uv run python -m src.approaches.llm.T2.experiment evaluate \
  --checkpoint models/T2_temporal/best.pt \
  --output data/T2/evaluation --batch-size 8 --device cpu --alpha 1
```

Produz `prediction.h5ad`, `target.h5ad`, `reference.h5ad`, `source_raw.h5ad`,
`manifest.json` e `metrics.json`. Usa `veckit.score(task="T1", input=prediction,
target=target, reference=reference)`: **T1 é a task temporal do veckit**, enquanto
T2 é o nome desta pasta. Não fornece rótulos futuros inventados ao avaliador.
A biblioteca pode inferi-los a partir da expressão predita.

Os arquivos originais são abertos em modo `backed`; somente as linhas amostradas
são materializadas. A avaliação recusa amostras acima de 2.000 e não passa os
arquivos completos ao veckit. Uma nova execução no mesmo diretório sobrescreve
os arquivos da avaliação. Reutilizar resultados para ajustar hiperparâmetros
consome a independência da reserva como teste final.

Inferência avulsa (a entrada deste comando deve estar previamente amostrada se
for usada para avaliação):

```bash
uv run python -m src.approaches.llm.T2.temporal predict \
  --checkpoint models/T2_temporal/best.pt \
  --input data/T2/evaluation/source_raw.h5ad --time 8.5 --delta-time 1.0 \
  --alpha 1 --output data/T2/prediction_again.h5ad
```

## Busca de hiperparâmetros na GPU

Com o manifesto e os recursos acima preparados, execute da raiz do projeto:

```bash
uv run python -m src.approaches.llm.T2.search
```

O script roda **seis treinos sequenciais** com CUDA, cinco épocas, até 512
células por estágio, 512 tokens e `batch-size=8` por padrão. A lista editável
`SEARCH_CONFIGS` em `search.py` varia taxa de aprendizado, número de blocos
ajustáveis, entrada de expressão projetada e peso de MMD. `--max-experiments`
limita o número de itens da lista; `--epochs`, `--max-cells`, `--max-length`,
`--batch-size`, `--alpha`, `--mode`, `--pairing`, `--ot-regularization`, `--device`
e os caminhos dos recursos são argumentos.
O comando `--help` mostra todos eles. Reduza `--batch-size` ou `--max-length`
se uma configuração exceder a VRAM. Cada falha fica registrada e a busca segue.

Cada iteração chama `temporal.train` e `experiment.evaluate`, incluindo a
predição E9.5 e `veckit.score(task="T1")`. O `results.json` de cada busca fica
em `models/T2_temporal/search/<data-hora>/` e é atualizado após cada iteração.
Ele contém hiperparâmetros, todas as métricas do Veckit, erros e o critério de
seleção. Os checkpoints e as avaliações individuais ficam em subdiretórios
`iteration_XX/`. Se já existir `models/T2_temporal/best.pt`, uma cópia é salva
como `previous_best.pt` antes da busca. Ao melhorar o resultado, o checkpoint
da iteração é copiado de forma atômica para `models/T2_temporal/best.pt`, o
caminho padrão do treino. Uma iteração pior não o substitui.

O critério é lexicográfico: **maior** `de_score`, depois **maior**
`de_direction`, depois **menor** `mmd_u` e, por fim, **menor** `variogram`.
Isso prioriza as métricas de mudança gênica do Veckit e usa distâncias como
desempate, sem ocultar os valores individuais. O código instalado do Veckit
define `de_score` como recuperação acima do nulo (1 corresponde ao conjunto
verdadeiro), `de_direction` como correlação de direção, `mmd_u` como distância
de distribuições e `variogram` como erro quadrático entre variogramas. A MMD
não enviesada pode ser ligeiramente negativa por ruído amostral; menor ainda é
melhor. A reserva E8.5/E9.5 passa a ser **validação para escolha de
hiperparâmetros** nesta busca. Para estimar desempenho final sem esse ajuste,
é necessária outra amostra independente.

## Experimentos controlados

Use o mesmo manifesto, seed, parâmetros de treino e orçamento em cada execução.
O comando de treino piloto acima fornece a base; troque somente `--output` e a
opção indicada abaixo. Para treinos completos, retire o limite piloto de 128
células e 3 épocas de forma **igual em todas as execuções**. Não use a reserva
E8.5/E9.5 para escolher hiperparâmetros; use a validação interna e avalie a
reserva apenas depois de fixar a comparação.

1. Treine a base com `--expression-input none --mmd-weight 0` e saída
   `models/T2_base`. Compare `α=0`, `0.5`, `1` e `2` no **mesmo** checkpoint,
   mudando só `--alpha` no comando `temporal predict`. `α=0` mede persistência.
2. Treine `models/T2_mmd` com `--expression-input none --mmd-weight 0.1`.
   Compare com a base usando `α=1`.
3. Treine `models/T2_expression` com `--expression-input projected` e
   `--mmd-weight 0`. Compare com a base usando `α=1`.
4. Se 2 e 3 melhorarem os critérios escolhidos na validação interna, treine
   `models/T2_combined` com `--expression-input projected` e
   `--mmd-weight 0.1`.

Para gerar as predições de cada `α`, execute, por exemplo:

```bash
uv run python -m src.approaches.llm.T2.temporal predict \
  --checkpoint models/T2_base/best.pt \
  --input data/T2/evaluation/source_raw.h5ad \
  --time 8.5 --delta-time 1 --alpha 0.5 \
  --output data/T2/alpha_05.h5ad
```

Para pontuar uma configuração já escolhida, use `experiment evaluate` com
`--checkpoint`, `--alpha` e `--output`; esse comando recupera a mesma
reserva e salva referência, alvo e predição alinhados. Repita com seeds
diferentes se a diferença for pequena.

Para comparar pareamentos, repita o mesmo treino com a mesma seed, `--mode`,
manifesto e demais opções, mudando só `--pairing ot --ot-regularization 0.1`
e `--output`. Comece com `--max-cells 128`: a matriz OT ocupa memória
proporcional a origens × destinos e é gravada para cada transição. Compare MSE
e MMD da validação interna antes de usar a reserva de avaliação. Repita com
outras seeds para medir a variação da amostragem. Veja a
[proposta de transporte ótimo](OT_PROPOSAL.md) para alternativas futuras.

## Testes e limites

```bash
uv run pytest -q tests/test_T2.py
```

Testes cobrem rank/medianas, congelamento, máscara, identidade, dependência de
`t` e `Δt`, intervalos irregulares, separação antes dos pares, reserva reproduzível,
limite de memória por amostra e treino/checkpoint/inferência com BERT minúsculo
aleatório **somente nos testes**.

Os destinos escolhidos são pseudo-pares, não descendentes observados. O modelo produz uma
previsão determinística por origem; não modela explicitamente destinos múltiplos,
proliferação ou morte. A avaliação por células mede distribuição E8.5→E9.5 em
estágios presentes no treino; não demonstra generalização para outro embrião ou
E10.5. O treino piloto é uma execução inicial, não um ajuste exaustivo.

## Execução realizada

Veja [resultados do piloto e auditoria](RESULTS.md), incluindo a comparação real
com veckit sobre 2.000 células por estágio.
