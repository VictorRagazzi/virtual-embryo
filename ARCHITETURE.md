# Arquitetura do projeto

Este documento é o primeiro ponto de consulta para entender o repositório. Ele descreve os scripts e as abordagens de predição baseadas em Transformers.

## Fluxo geral

O projeto tenta prever a expressão gênica de células do estágio E10.5 a partir dos dados observados em E8.5 e E9.5.

O fluxo esperado é linear:

1. ler os arquivos `.h5ad`;
2. inspecionar e preparar os dados;
3. executar uma abordagem de predição;
4. gerar um novo `.h5ad`;
5. comparar ou pontuar o resultado.

Os dados ficam em `data/`, as figuras em `figs/`, os modelos em `models/` e o código em `src/`.

## Scripts

`src/scripts/` contém programas executáveis e independentes. Cada script deve representar uma operação completa, com argumentos de linha de comando, etapas na ordem de execução e saída explícita. Eles não formam um serviço e não precisam de camadas extras.

| Script | Responsabilidade |
| --- | --- |
| `visualize.py` | Inspeciona um `.h5ad` e gera resumos e gráficos exploratórios. |
| `plot_umap.py` | Calcula e desenha UMAP, com correção Harmony opcional. |
| `pre_processement.py` | Valida, amostra e filtra genes; também pode agrupar genes correlacionados. |
| `subsample.py` | Reduz o número de células preservando a proporção dos tipos celulares. |
| `predict_e10_5_PCA.py` | Baseline completo: calcula PCA conjunto, extrapola mudanças de E8.5 para E9.5 e monta a previsão de E10.5. |
| `subset_genes.py` | Cria subconjuntos compatíveis e menores para testar a pontuação local. |
| `compare.py` | Compara dois arquivos `.h5ad` e destaca diferenças globais, por gene e por tipo celular. |
| `test_score.py` | Executa uma validação simples do pipeline de pontuação. |
| `get_dummy.py` | Rascunho desativado para gerar ou amostrar uma submissão de teste. |
| `test.py` | Arquivo vazio, sem responsabilidade definida no momento. |
| `split_by_day.py` | Separa um `.h5ad` por uma coluna temporal, preserva expressão e gera um manifesto sem sobrescrever saídas. |
| `subset_heart.py` | Seleciona rótulos cardíacos explícitos em E85/E95 e no índice Hugging Face; preserva a expressão e registra a origem do dia informado. |

Ao adicionar uma etapa curta, prefira incluí-la no script ao qual ela pertence. Crie outro arquivo somente quando existir uma operação independente que faça sentido executar sozinha.

## Abordagem LLM

O código experimental fica em `src/approaches/llm/T1/`. Apesar do nome da pasta, o fluxo principal usa o scGPT, um Transformer treinado para representar expressão gênica. Há duas etapas conectadas:

### 1. Pareamento temporal com Optimal Transport

`moscot_apply/` aproxima quais células de E8.5 correspondem a células de E9.5:

1. `data_utils.py` carrega os dois estágios, alinha os genes e calcula uma PCA conjunta;
2. `ot_pairing.py` calcula a matriz de transporte com MOSCOT;
3. `pairing_sampler.py` converte os pesos em pares de células;
4. `build_dataset.py` salva os pares e a matriz de transporte.

O resultado serve como dados de treino para a etapa seguinte. O pareamento é uma aproximação probabilística, não um rastreamento real de linhagem celular.

### 2. Ajuste e predição com scGPT

`fine_tuning/` aprende a transformação E8.5 → E9.5 e aplica a mesma ideia sobre E9.5 para estimar E10.5:

1. `config.py` reúne caminhos e hiperparâmetros;
2. `vocab_utils.py` seleciona genes presentes no vocabulário do scGPT;
3. `data_prep.py` transforma os pares em entradas e alvos;
4. `scgpt_model.py` define o modelo e adapta os pesos pré-treinados;
5. `train_finetune.py` treina o regressor;
6. `predict.py` produz a previsão final e recompõe genes não previstos.

Os `README.md` dentro de `moscot_apply/` e `fine_tuning/` guardam comandos, parâmetros e limitações específicos. Consulte-os apenas ao trabalhar diretamente nessas etapas.

## Mouse-Geneformer temporal

`src/approaches/mouse_geneformer/` implementa uma abordagem independente de `llm/T1`, com múltiplos estágios e intervalos explícitos:

1. `data.py` alinha genes, normaliza expressão, cria tokens por rank/mediana e separa células; `select_future_neighbors` concentra a estratégia substituível de pareamento;
2. `temporal.py train` ajusta uma SVD somente no treino, cria pseudo-pares dentro de cada split e treina todas as transições crescentes informadas;
3. `model.py` congela o encoder pré-treinado, exceto os últimos blocos configuráveis, e prevê expressão direta ou delta com `t` e `Δt`;
4. `temporal.py predict` usa o checkpoint para exportar expressão futura em escala log1p normalizada.

Os dados embrionários locais confirmados são E8.5/E9.5. Os arquivos baixados de Weinreb e Veres foram separados em `data/by_day/`, mas seus dias experimentais não são estágios embrionários. O índice Hugging Face referencia outras matrizes que não estão disponíveis localmente. Pesos, vocabulário, medianas, mapeamento de genes e confirmação da escala de entrada são requisitos para o treino real. Veja [comandos, dados e limitações](src/approaches/mouse_geneformer/README.md).

## Experimento Mouse-Geneformer em T2

`src/approaches/llm/T2/` adapta a implementação local de Mouse-Geneformer para
o experimento multitemporal com os arquivos embrionários disponíveis E6.75–E9.5.
`experiment.py prepare` reserva IDs de 2.000 células oficiais de E8.5 e de E9.5.
`temporal.py train` exclui a reserva antes da preparação e do pareamento;
`data.py::pair_cells` concentra a escolha do vizinho em uma projeção ajustada
somente no treino. O modelo recebe o tempo de origem e o intervalo de cada par.
`experiment.py evaluate` recupera a reserva do checkpoint, prevê E9.5 e chama
`veckit.score(task="T1")` somente com as amostras. O código experimental anterior
foi preservado. Veja [execução e limitações](src/approaches/llm/T2/README.md).

## Regra para novas implementações

Mantenha o caminho principal visível em uma leitura de cima para baixo. Prefira poucas funções maiores, com responsabilidades claras, a muitas funções pequenas espalhadas por arquivos. Separe código apenas quando houver uma fronteira concreta: uma etapa executável independente, uma responsabilidade reutilizada ou uma parte que precise ser testada isoladamente.
