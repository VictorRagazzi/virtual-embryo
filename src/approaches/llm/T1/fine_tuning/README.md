# Fine-tuning scGPT — Task 1 (Temporal)

Pipeline de 3 passos, seguindo o pareamento OT que você já calculou:

1. `train_finetune.py` — carrega o scGPT pré-treinado (`models/scGPT_heart/best_model.pt`),
   monta os pares E8.5 → E9.5 a partir do plano de transporte, e faz fine-tuning
   de regressão (previsão de expressão, no mesmo espírito do perturb-GEP do scGPT).
2. `predict_e10.py` — aplica o modelo treinado sobre E9.5 para gerar a predição de E10.5.
3. `config.py` centraliza caminhos e hiperparâmetros — ajuste ali, não nos outros arquivos.

## Antes de rodar

- **`vocab.json` do scGPT precisa estar em `models/scGPT_heart/vocab.json`.**
  Se você baixou só o `best_model.pt` + `args.json`, pegue o `vocab.json`
  correspondente no repositório do scGPT (é o mesmo vocabulário citado em
  `args.json`, `default_census_vocab.json`).
- Dependências (via uv):
  ```
  uv add torch scanpy anndata scipy numpy
  ```

## Como rodar

Da raiz do projeto:
```
uv run python -m src.approaches.llm.T1.fine_tuning.train_finetune
uv run python -m src.approaches.llm.T1.fine_tuning.predict
```

(Cada pasta entre `src/` e `fine_tuning/` precisa ter um `__init__.py`, mesmo
vazio, para o `-m` funcionar.)

## Decisões de simplificação, para justificar no relatório

- **Transformer**: o checkpoint foi treinado com `fast_transformer` (flash-attention),
  mas as chaves salvas (`self_attn.Wqkv`, `norm1`, `norm2`, `linear1/2`) são
  equivalentes às do `nn.TransformerEncoderLayer` padrão do PyTorch — só é
  preciso renomear `Wqkv` para `in_proj_weight/bias`. Isso evita instalar a
  biblioteca `flash-attn` (pesada e dependente de CUDA compilado).
- **Tokenização por gene**: cada gene vira um "token" (id do vocabulário) + um
  valor de expressão (binned em 51 faixas), igual no scGPT original.
- **Limite de 1200 genes por célula** (`max_seq_len` do modelo): usamos os
  1200 genes de maior variância que também existem no vocabulário do scGPT.
- **Pareamento**: para cada célula de E8.5, usamos a célula de E9.5 com maior
  peso na linha correspondente do plano de transporte (pareamento 1-para-1,
  "duro"). É a forma mais simples de transformar o plano de OT (probabilístico)
  em pares de treino.
- **Alvo do treino**: a expressão contínua real (não binned) da célula pareada
  de E9.5 — o decoder do scGPT (`decoder.fc`) já prevê um valor contínuo por gene.
- **`flag_encoder`**: no scGPT original ele marca "valor mascarado vs. valor
  dado". Como não fazemos mascaramento (é regressão direta t → t+1), sempre
  usamos a flag "dado" (índice 0).
- **Genes fora do limite de 1200**: o scGPT só prevê os genes do `gene_list`
  (os 1200 selecionados). Para todos os outros genes que sobreviveram ao
  filtro de variância mas não entraram no modelo, `predict_e10.py` copia o
  valor de E9.5 direto (função `fill_missing_genes` em `data_prep.py`) — ou
  seja, assumimos que, sem uma predição do modelo, o mais razoável é dizer
  que esse gene não muda de E9.5 para E10.5. A predição final sai com o
  mesmo conjunto de genes do arquivo `E95.h5ad` usado (não só os 1200).
  Se em algum momento vocês precisarem devolver os ~19.355 genes cortados
  no filtro de variância original (para bater com os 32.285 genes da
  competição), basta chamar `fill_missing_genes` de novo, passando como
  referência o E9.5 *não filtrado* (antes do `pre_processement.py`).

## Se o carregamento dos pesos falhar (muitos `missing_keys`)

O script imprime quantos pesos foram carregados e quantos ficaram faltando.
Se o número de "faltando" for muito alto, provavelmente os nomes internos das
camadas do checkpoint são diferentes do que assumimos aqui. Nesse caso, rode:

```python
import torch
sd = torch.load("models/scGPT_heart/best_model.pt", map_location="cpu")
for k in list(sd.keys())[:20]:
    print(k, sd[k].shape)
```

e ajuste a função `_remap_key` em `scgpt_model.py` de acordo com os nomes reais.