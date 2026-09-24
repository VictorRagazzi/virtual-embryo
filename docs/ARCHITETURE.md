# Arquitetura proposta

## 1. Visão geral

```text
E8.5 / E9.5 (.h5ad)
        │
        ▼
normalização validada e alinhamento de genes
        │
        ▼
scGPT auditado e inicialmente congelado
        │
        ▼
embeddings scGPT por célula
        │
        ▼
agrupamento conjunto + resumos por grupo e estágio
        │
        ▼
Transformer temporal numérico condicionado pelo tempo alvo
        │
        ▼
proporções, centros e dispersões dos grupos futuros
        │
        ▼
amostrador de células latentes
        │
        ▼
decoder
        │
        ▼
matriz de expressão [células × 32.285 genes]
        │
        ▼
validação + arquivo .h5ad
```

## 2. Princípios

### 2.1 O espaço principal é o embedding do scGPT

O único embedding celular usado pela nova abordagem será o do scGPT. Não haverá combinação com Mouse Geneformer nem um segundo encoder celular na primeira versão.

Como o embedding scGPT não é automaticamente invertível, será treinado um decoder separado:

```text
expressão real → scGPT → embedding → decoder treinado → expressão reconstruída
```

O teste de round-trip decide se essa representação contém informação suficiente. Se falhar, o agente deve registrar o bloqueio e apresentar evidências; não deve introduzir silenciosamente outro encoder.

### 2.2 A unidade temporal é a população

O Transformer não recebe uma célula e tenta encontrar sua sucessora. Ele recebe estados/grupos que resumem a população em cada estágio.

### 2.3 A geração preserva diversidade observada

O gerador inicial não cria células inteiramente do zero. Ele combina centros futuros previstos com resíduos/variações retirados de células reais ou de um gerador explicitamente treinado.

## 3. Contratos entre componentes

Os nomes abaixo são contratos conceituais. Dimensões exatas devem ficar em configuração.

### 3.1 Dados de expressão

```text
expression: float32 [n_cells, 32285]
gene_names: string [32285]
stage: float [n_cells]
```

Invariantes:

- mesma ordem de genes em todos os estágios;
- valores finitos e não negativos;
- escala log-normalizada confirmada;
- dados brutos não são alterados.

### 3.2 Espaço latente celular scGPT

```text
cell_latent: float32 [n_cells, latent_dim]
```

Configuração inicial:

- `latent_dim`: determinada pelo checkpoint scGPT escolhido;
- scGPT inicialmente congelado;
- decoder treinado para mapear o embedding scGPT aos 32.285 genes;
- saída do decoder não negativa;
- batches e leitura esparsa compatíveis com a memória disponível.

### 3.3 Atribuição a grupos

```text
group_id: int [n_cells]
ou
group_probability: float32 [n_cells, n_groups]
```

Primeira versão:

- agrupamento conjunto de E8.5 e E9.5;
- K-means como baseline;
- testar `n_groups` em uma pequena grade, inicialmente 50, 100 e 200;
- Leiden ou mistura probabilística apenas após o baseline.

### 3.4 Resumo de um grupo em um estágio

Cada token de população deve conter, no mínimo:

```text
stage
group_identifier
proportion
latent_mean
latent_dispersion
gene_program_scores
```

Campos de diagnóstico, não obrigatoriamente enviados ao Transformer:

```text
cell_count
mean_expression_full
top_up_genes
top_down_genes
quality_flags
```

Os top genes servem para interpretação. Eles não substituem a representação do transcriptoma completo.

### 3.5 Entrada do Transformer

Para `K` grupos e dois estágios:

```text
population_tokens: [2*K, population_feature_dim]
target_time: scalar ou embedding temporal
```

Uma camada linear converte cada token para `model_dim`.

Configuração inicial sugerida:

- encoder-decoder numérico;
- 4 camadas;
- `model_dim` 256;
- 8 cabeças de atenção;
- dropout configurável;
- sem dependência de vocabulário textual.

### 3.6 Saída do Transformer

Primeira versão com dicionário de grupos fixo:

```text
future_proportion: [K]
future_latent_mean: [K, latent_dim]
future_latent_dispersion: [K, latent_dim]
future_program_scores: [K, n_programs]
```

As proporções devem ser não negativas e somar 1. Grupos ausentes podem receber massa próxima de zero.

Uma versão posterior pode usar slots futuros livres, desde que haja um procedimento explícito para casar previsões e grupos reais durante o treinamento.

## 4. scGPT e decoder

### 4.1 Objetivo inicial

Aprender:

```text
x_real → scGPT congelado → z_scgpt → decoder → x_reconstruído
```

O teste principal é populacional, não apenas erro médio por célula. Comparar a população reconstruída com a real usando estatísticas alinhadas à competição.

### 4.2 Perdas candidatas

Começar simples e adicionar apenas quando necessário:

- reconstrução por gene;
- preservação de pseudobulk;
- preservação de variância;
- preservação de covariação em pares amostrados;
- regularização do decoder.

Não otimizar todas simultaneamente sem ablação.

## 5. Agrupamento e resumos

O agrupamento é ajustado sobre as células concatenadas para criar um vocabulário compartilhado. Para cada par `(estágio, grupo)`, calcular:

- número e proporção de células;
- média latente;
- dispersão latente;
- atividade de programas gênicos;
- expressão média completa para avaliação;
- resíduos individuais `z_i - média_do_grupo` para geração posterior.

Grupos vazios em um estágio continuam no vocabulário com massa zero e máscara apropriada.

## 6. Transformer temporal

O Transformer aprende uma função:

```text
populações observadas + tempo alvo → população futura
```

Ele deve ser treinado em tarefas cujo alvo é conhecido, preferencialmente usando séries externas permitidas e separação temporal explícita.

Perdas iniciais:

- erro nas proporções;
- erro nos centros latentes;
- erro nas dispersões;
- erro nos programas gênicos;
- perdas populacionais calculadas após geração, quando o pipeline permitir.

## 7. Geração de células

### 7.1 Gerador inicial por resíduos empíricos

Para gerar uma célula do grupo `k`:

```text
z_novo = centro_futuro[k] + escala_futura[k] * residuo_amostrado[k]
```

Onde `residuo_amostrado[k]` vem de uma célula real do grupo, centralizada pelo centro do estágio de origem.

Depois:

```text
x_novo = decoder(z_novo)
```

Vantagens:

- preserva variação observada;
- reduz colapso para centroides;
- fornece um baseline interpretável;
- não exige um segundo gerador complexo na primeira versão.

### 7.2 Grupos novos ou sem resíduos

Ordem de fallback:

1. resíduos do grupo mais próximo no espaço latente;
2. resíduos de estado semelhante em dados externos permitidos;
3. gerador condicional treinado, em uma versão posterior.

Todo fallback deve ser marcado nos artefatos para permitir auditoria.

## 8. Uso do scGPT

Testar em sequência:

1. scGPT congelado + decoder simples;
2. scGPT congelado + decoder com perdas populacionais;
3. ajuste parcial do scGPT apenas se a proveniência permitir e houver evidência de necessidade.

Não introduzir outro foundation model. Se o decoder não reconstruir adequadamente a partir do embedding scGPT, registrar o resultado antes de alterar a arquitetura.

## 9. Configuração e artefatos

Cada execução deve salvar:

- configuração serializada;
- seed;
- hashes ou identificadores dos dados;
- checkpoint;
- métricas em JSON/CSV;
- figuras de diagnóstico;
- predição `.h5ad`, quando aplicável;
- ID do experimento.

Sugestão de diretórios novos, sem exigir reorganização do código existente:

```text
configs/population_transformer/
src/approaches/population_transformer/
outputs/population_transformer/<experiment_id>/
tests/population_transformer/
```
