# Arquitetura do projeto

Este documento é o primeiro ponto de consulta para entender o repositório. Ele descreve apenas as duas áreas que mais precisam de contexto: os scripts e a abordagem baseada em modelos de linguagem.

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

## Regra para novas implementações

Mantenha o caminho principal visível em uma leitura de cima para baixo. Prefira poucas funções maiores, com responsabilidades claras, a muitas funções pequenas espalhadas por arquivos. Separe código apenas quando houver uma fronteira concreta: uma etapa executável independente, uma responsabilidade reutilizada ou uma parte que precise ser testada isoladamente.
