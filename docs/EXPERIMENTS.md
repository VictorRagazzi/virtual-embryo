# Registro de experimentos

## Como usar

1. Reserve o próximo ID antes da execução.
2. Declare hipótese e critério de decisão antes de observar resultados.
3. Registre inclusive resultados negativos.
4. Nunca sobrescreva resultados anteriores; acrescente uma nova entrada.

## Índice

| ID | Data | Marco | Hipótese | Configuração | Resultado principal | Decisão | Estado |
|---|---|---|---|---|---|---|---|
| E001 | 2026-09-23 | M0 | Os baselines estruturais e o scorer local são reproduzíveis sem densificar os dados completos | seed 42; auditoria em blocos; PCA limitada a 128 células/estágio | contrato passou; `copy_last` integral e pseudoavaliação `veckit` executados | encerrar M0; checkpoint scGPT bloqueado por proveniência | CONCLUÍDO |
| E002 | 2026-09-23 | M1 | O checkpoint local pode ser liberado após auditoria de proveniência | SHA-256 local; busca de fonte; sem download/treino | hash identificou fonte, mas corpus/estágios permanecem indeterminados | bloquear treino/uso oficial | BLOQUEADO |
| E003 | pendente | M2 | K=100 equilibra detalhe e estabilidade | `K={50,100,200}` | pendente | pendente | PLANEJADO |
| E004 | pendente | M3 | Centros + resíduos empíricos preservam diversidade | pendente | pendente | pendente | PLANEJADO |
| E005 | 2026-09-23 | M1 | Um checkpoint oficial humano pode ser uma alternativa auditável a M001 | leitura de metadados oficiais; sem download | candidato registrado, mas sem objeto/version/hash ou manifesto temporal | manter PENDENTE; não baixar/usar | CONCLUÍDO |
| E006 | 2026-09-24 | M1 | O round-trip exploratório com M001 é permitido para a disciplina | autorização explícita; sem submissão | M001 liberado apenas para experimento | iniciar extração/decoder M1 | AUTORIZADO |
| E007 | 2026-09-24 | M1 | Decoder linear scGPT supera o perfil médio | 128 células; 256 genes; 8 épocas | MSE melhora 5,0%, mas variância colapsa a 6,75% | rejeitar como decoder final | CONCLUÍDO_NEGATIVO |
| E008 | 2026-09-24 | M1 | MLP + perda de variância recupera diversidade | mesmo split; MLP 512; peso 5 | variância 26,6%, mas MSE/covariância pioram | rejeitar | CONCLUÍDO_NEGATIVO |
| E009 | 2026-09-24 | M1 | Perda de variância moderada equilibra MSE/diversidade | mesmo split; linear; peso 1 | MSE melhora 3,6%, variância ainda 12,8% | rejeitar; manter M1 | CONCLUÍDO_NEGATIVO |
| E010 | 2026-09-24 | M1 | Mais genes de entrada recuperam diversidade do decoder linear | mesmo split; 512 genes; sem perda extra | MSE melhora 5,0%, variância cai a 6,09% | rejeitar; cobertura não é o gargalo principal | CONCLUÍDO_NEGATIVO |
| E011 | 2026-09-24 | M1 | Perda de covariância em genes variáveis reduz colapso sem perder o ganho de MSE | mesmo split; linear; covariância normalizada, peso 0,1 em 128 genes | MSE melhora 4,4%, variância sobe só a 7,60% | rejeitar; manter M1 e M2 bloqueado | CONCLUÍDO_NEGATIVO |
| E012 | 2026-09-24 | M1 | Projeção da saída em referência E9.5 recupera diversidade | E011 + vizinho de expressão E9.5, incluindo validação | variância 39,5%, mas MSE 0,09527 e só 6 referências | diagnóstico vazado; não usar como decoder | CONCLUÍDO_DIAGNÓSTICO_VAZADO |
| E013 | 2026-09-24 | M1 | Resíduo conhecido da validação confirma o teto estrutural do caminho completo | E011 + resíduo exato da validação | MSE ≈ 0; variância/covariância ≈ 1 | autorizar M2 exploratório por decisão do responsável; não é evidência de decoder | CONCLUÍDO_DIAGNÓSTICO_VAZADO |
| E014 | 2026-09-24 | M2 | Um vocabulário K-means conjunto em CLS scGPT produz tokens populacionais reproduzíveis e K=50/100/200 permite quantificar o compromisso entre estabilidade e grupos pequenos/vazios | D001–D011; CLS 512; 256 genes; seeds 42/43/44; `K={50,100,200}` | K=50 teve maior ARI médio (0,425), menor L1 de proporções e 340 grupos estágio×cluster vazios | adotar K=50 como baseline exploratório; estabilidade moderada exige cautela | CONCLUÍDO |
| E015 | 2026-09-24 | M3 | Centros + resíduos empíricos compatíveis com E011 preservam mais diversidade que repetir centroides ao reconstruir E9.5 conhecido | K=50; seed 42; tokenização exata E011; `n={128,512,2048}` e decodificação principal de 512 células | 506 células únicas e variância 0,0772 vs. 45/0,0704 nos centroides, mas pseudobulk/MMD não melhoraram | confirmar gerador latente; bloquear avanço porque E011 colapsa após decodificação | CONCLUÍDO_NEGATIVO |
| E016 | 2026-09-24 | M3/M4 | Resíduos de expressão conhecidos de E8.5 transferidos por grupo para uma reconstrução E9.5 recuperam diversidade sem usar resíduos E9.5 | K=50; seed 42; decoder E011; 512 células por estágio | variância 0,9517 e RFF-MMD 0,0106 vs. 0,0771/0,0819 no decoder cru; 1 fallback | manter correção por resíduo conhecido como variante principal exploratória | CONCLUÍDO |
| E017 | 2026-09-24 | M4/M5 | Copy-last e extrapolação linear estabelecem referências temporais reproduzíveis no holdout E8.5 | K=50; seed 42; D009/D010→D011 | copy-last agregado 0,00372; linear 0,16389, mas linear melhora proporções | comparar E018 sob a mesma avaliação | CONCLUÍDO |
| E018 | 2026-09-24 | M5 | Um Transformer pequeno aprende dinâmica de tokens e melhora o erro agregado no holdout E8.5 contra os baselines | dim 128; 2 camadas; 4 cabeças; 250 épocas; seed 42 | Transformer 0,01976; MLP 0,00941; ambos perdem para copy-last 0,00372 | registrar resultado negativo; pipeline é operacional e pode extrapolar exploratoriamente | CONCLUÍDO_NEGATIVO |
| E019 | 2026-09-24 | M5/M7 | O Transformer válido pode ser reajustado nos estágios conhecidos e produzir E10.5 experimental com três variantes estruturalmente válidas | E6.5–E8.5 para dinâmica; E8.5/E9.5 oficiais como entrada; 512 células; seed 42 | três `.h5ad` válidos; residual E9.5 tem variância 0,882 vs. 0,041 crua em relação a E9.5 | concluir pipeline; manter não submetível e recomendar residual somente como variante exploratória | CONCLUÍDO_EXPLORATÓRIO |
| E020 | 2026-09-24 | M5/M7 | Losses diferenciáveis alinhadas a DE/ranking e RFF-MMD, com seleção relativa ao copy-last, podem tornar a busca do Transformer mais informativa sem usar E10.5 | smoke sintético: 24 células, 32 genes, 2 épocas, seed 42; servidor: 2.048 genes e busca ≤12 runs | smoke completo passou; loss 0,41341, gradientes finitos e três contratos 24×32 válidos | componentes validados; integração pesada real ainda depende do servidor e dos caches | CONCLUÍDO_SMOKE |
| E021 | 2026-09-24 | M5/M7 | Ancorar a mudança prevista em células reais do último estágio pode preservar esparsidade/covariação sem abandonar o Transformer temporal | 32 programas; `α_expr=α_prop=0,25`; seed 42; 1.024 células; 32.285 genes | LB: 38,8/51,4/48,4/48,5; ponderado 46,77 vs 41,21 no E020 | validar Estratégia A; próximo ciclo deve focar DE sem regredir distribuição/variograma | CONCLUÍDO_POSITIVO |
| E022 | 2026-09-24 | M5/M7 | Loss up/down em múltiplos cortes melhora recuperação dos genes DE sem regredir distribuição/covariação | top-50/100/250/500/1.000; pesos 0,05/0,1/0,2; seed 42; 512 células | melhor DE 0,0190 vs 0,0476 no E021; MMD/variograma também regrediram | rejeitar loss atual; manter candidata E021 | CONCLUÍDO_NEGATIVO |

| E023 | 2026-09-24 | M5/M7 | Programas híbridos temporais ampliam recuperação DE | 32+16 e 32+32; seeds 42; 512 células | DE 0,0286/0,0095 vs 0,0476 E021; distribuição pior | rejeitar; parar por gates; E021 preservado | CONCLUÍDO_NEGATIVO |
| E024 | 2026-09-24 | M5/M7 | Tendência gênica recente simples melhora DE sem regredir as demais métricas | pesos 0,0625/0,125/0,25; 512→1.024 células | ganho DE, mas direção falhou em 1.024 | rejeitar | CONCLUÍDO_NEGATIVO |
| E025 | 2026-09-24 | M5/M7 | Restringir a tendência aos genes mais fortes preserva direção | top-250/1.000; seeds 42/43/44 | DE melhorou nas três seeds; gates conjuntos só na 42 | não substituir candidata por instabilidade | CONCLUÍDO_PARCIAL |
| E026 | 2026-09-25 | M5/M7 | Reduzir a loss estrutural prioriza a saída usada na geração | peso estrutural 0,1; seed 42; 512 células | DE 0,0762, mas direção/MMD/variograma regrediram | rejeitar e não confirmar em 1.024 | CONCLUÍDO_NEGATIVO |
| E027 | 2026-09-25 | M5/M7 | Kernel PCA e células cruas preservam informação melhor que scGPT e resumos de grupo | Nyström-RBF+PCA 32D; 512 genes; agrupado vs. 128 células cruas; três decoders; seed 42 | células cruas+resíduo: DE 0,0952, direção 0,0576, MMD 0,12592, variograma 0,007172 | rejeitar; E021 permanece candidata | CONCLUÍDO_NEGATIVO |
| E028 | 2026-09-25 | M5/M7 | Remover `_ex` e aumentar o Transformer melhora a transição oficial conhecida | apenas D001/D002; split celular 1.024/256/512; 71k/538k/3,18M parâmetros | desempenho piorou monotonicamente com a complexidade; mean shift venceu | não aumentar modelo; sem evidência temporal para remover `_ex` | CONCLUÍDO_DIAGNÓSTICO |
| E029 | 2026-09-25 | M5/M7 | Pares de grupos K-means E8.5→E9.5 capturam a mudança melhor que células aleatórias | K=32 conjunto; somente D001/D002; 77k/551k/3,20M parâmetros | delta direto por grupo venceu; Transformer pequeno foi o melhor neural; escala piorou | validar desenho de grupos, manter modelo pequeno; não alegar E10.5 | CONCLUÍDO_DIAGNÓSTICO |
| E030 | 2026-09-25 | M5/M7 | Âncoras de expressão esparsas recuperam variograma das candidatas E029 | delta/small/medium; escalas 0,125/0,25; 2.500 células | zeros restaurados de 60–61% para 87,7–87,9%; seis candidatas válidas | submeter sempre os três modelos sob a mesma escala | AGUARDA_LEADERBOARD |
| E031 | 2026-09-25 | M5/M7 | Ativação seletiva por prevalência recupera sinal temporal sem redensificar a população | delta/small/medium; 512 células; seed 42; teto 256 ativações/célula | melhorou direção, MMD e variograma nos três; DE melhorou em dois e empatou em um | gates passaram; trio de 2.500 gerado | CONCLUÍDO_LOCAL |
| E032 | 2026-09-25 | M5/M7 | Reduzir apenas a mudança do decoder em zeros recupera esparsidade sem abandonar a geometria E029 | delta/small/medium; beta 0,10/0,20; 512 células; seed 42 | variograma melhorou 64–77% e MMD foi preservado/melhorado; beta 0,20 falhou direção do delta por 2,1 pp do gate | resultado parcial; não gerar 2.500 | CONCLUÍDO_PARCIAL |
| E033 | 2026-09-25 | M5/M7 | Transições `_ex` melhoram a dinâmica agrupada e a grade PCA×K revela o compromisso de capacidade | controle PCA=32/K=32; PCA={16,32,64}×K={16,32,64}; trio obrigatório | melhor `_ex`: PCA64/K64 group_delta 0,3710/0,4109/0,02012/0,000915; Transformers não superaram official-only | registrar mudança de domínio; não gerar candidata | CONCLUÍDO_NEGATIVO |
| E034 | 2026-09-25 | M5/M7 | A grade PCA×K official-only encontra capacidade melhor sem mudança de domínio | PCA={16,32,64}×K={16,32,64}; trio; beta 0,20; seed 42 | small PCA32/K32: 0,5323/0,4296/0,01852/0,000934; melhor compromisso | preservar checkpoint; candidata final ainda não gerada | CONCLUÍDO_POSITIVO |

## Template detalhado

### E001 — Auditoria, contrato e baselines M0

**Data:** 2026-09-23

**Marco:** M0

**Responsável/agente:** Codex

**Hipótese:** Os dados oficiais podem ser lidos sem alteração, têm o mesmo painel ordenado de genes e os baselines de formato/`copy_last` são executáveis de forma reprodutível no ambiente atual.

**Critério de decisão definido antes da execução:** salvar auditoria versionável, confirmar automaticamente 32.285 genes na mesma ordem, gerar e validar `copy_last`, e obter ao menos uma saída numérica do `veckit` sem exceder a memória por densificação.

**Código:** `5be5ecb` como revisão inicial; mudanças M0 não commitadas durante a execução. Estado inicial do Git limpo.

**Dados:** D001 e D002. E8.5 SHA-256 `8eab2d0ccaa89f92861b09816b76b6a731b8ed6b5995e4af196d9ed8ff76e504`; E9.5 SHA-256 `0296e0043842f944a663f3c11ac1542d1bbee733c1cd657bd56f75af65958361`. Nenhum dado externo, download, API ou Ollama foi usado.

**Configuração:**

```yaml
seed: 42
audit:
  x_scan_block_values: 5000000
  dense_matrix: false
copy_last:
  source: data/E95.h5ad
  target_cells: null
dummy:
  reference: data/E95.h5ad
  n_cells: 2500
pca_delta_centroid:
  cells_per_stage: 128
  n_components: 30
  skip_llm: true
veckit:
  task: T1
  pseudo_target: stratified E9.5 sample, 128 cells
  pseudo_reference: stratified E8.5 sample, 128 cells
```

**Comandos:**

```bash
uv run python src/scripts/audit_m0.py --seed 42
uv run python src/scripts/get_dummy.py --reference data/E95.h5ad --output /tmp/ve-m0/dummy.h5ad --n-cells 2500 --seed 42
uv run python src/scripts/copy_last.py --input data/E95.h5ad --output /tmp/ve-m0/copy_last.h5ad --seed 42
uv run python src/scripts/predict_e10_5_PCA.py --e85 data/E85.h5ad --e95 data/E95.h5ad --out /tmp/ve-m0/pca_e105_sample128.h5ad --max-cells-per-stage 128 --n-comps 30 --target-cells 1000 --skip-llm --seed 42
uv run python src/scripts/subsample.py --input-h5ad data/E85.h5ad --output-h5ad /tmp/ve-m0/e85_target128.h5ad --n-cells 128 --seed 42 --min-per-group 1
uv run python src/scripts/subsample.py --input-h5ad data/E95.h5ad --output-h5ad /tmp/ve-m0/e95_target128.h5ad --n-cells 128 --seed 42 --min-per-group 1
uv run veckit --task T1 --input /tmp/ve-m0/copy_last_target128.h5ad --target /tmp/ve-m0/e95_target128.h5ad --reference /tmp/ve-m0/e85_target128.h5ad --seed 42
uv run veckit --task T1 --input /tmp/ve-m0/pca_e105_sample128.h5ad --target /tmp/ve-m0/e95_target128.h5ad --reference /tmp/ve-m0/e85_target128.h5ad --seed 42
uv run pytest tests/test_m0_contract.py -q
```

**Métricas:**

| Métrica | `copy_last` (identidade do pseudoalvo) | PCA + delta (amostra 128) | Observação |
|---|---:|---:|---|
| `de_score` | 0.5882 | 0.5294 | pseudoalvo E9.5, não score de E10.5 |
| `de_direction` | 1.0000 | 0.4960 | identidade é controle de funcionamento, não previsão futura |
| `energy_distance` | -0.97966 | 1.08412 | estimador amostral pode ser negativo |
| `mmd_u` | -0.00956 | 0.05147 | estimador não enviesado pode ser negativo |
| `variogram` | 0.000000 | 0.004970 | |
| `pb_rel_err` | 0.0000 | 0.1177 | |
| `variance_ratio` | 1.0000 | 0.7240 | |
| `composition_JSD` | 0.0000 | 0.0324 | probe treinada no pseudoalvo rotulado |
| `pseudobulk_pearson` | 1.0000 | 0.9920 | |

**Artefatos:**

- relatório: `docs/M0_AUDIT.md` e `docs/M0_AUDIT.json`;
- temporários não versionados: `/tmp/ve-m0/*.h5ad`, `/tmp/ve-m0/*.json`;
- nenhum checkpoint, predição pesada ou dado bruto adicionado ao Git.

**Resultado:** E8.5 é CSC `float32` 16.787 × 32.285 (12,067% de valores efetivamente não zero); E9.5 é CSC `float32` 17.057 × 32.285 (12,221%). Ambos são finitos e não negativos, e os genes têm digest ordenado idêntico `807549e15be019899fa83cbab864bec87ad278f60c272e39819c7302cc2285b8`. Não há metadado suficiente para confirmar a normalização. O checkpoint scGPT local tem embedding 512, mas foi bloqueado por proveniência.

**Limitações:** `veckit` densifica todas as entradas; somente E8.5 + E9.5 completos já exigiriam 8,14 GiB em `float64`, antes de cópias e da predição. Por isso a pseudoavaliação usa 128 células. A PCA também usa 128 células por estágio e sua saída de 128 células fica abaixo do mínimo de submissão; ela é diagnóstico de execução, não candidata a submissão. A suíte completa ainda falha na coleta porque módulos históricos referenciados por `tests/test_mouse_geneformer.py` e `tests/test_subset_heart.py` não estão na árvore.

**Decisão:** manter os utilitários M0; não usar o checkpoint atual em submissão; não interpretar as métricas de E9.5 como desempenho em E10.5.

**Próximo experimento:** após autorização de M1, registrar e auditar um checkpoint scGPT de proveniência verificável antes de extrair embeddings; então medir o round-trip com um decoder mínimo, sem iniciar agrupamento ou Transformer.

### E002 — Proveniência do checkpoint scGPT heart

**Data:** 2026-09-23

**Marco:** M1

**Responsável/agente:** Codex

**Hipótese:** O hash do checkpoint local permite identificar sua fonte e determinar se seus dados de treinamento excluem a janela proibida.

**Critério de decisão definido antes da execução:** liberar somente se a identidade, licença e corpus/intervalo de estágios permitirem excluir E10.5–E13.5; caso contrário registrar `BLOQUEADO_POR_PROVENIENCIA` e não extrair embeddings nem treinar decoder.

**Código:** árvore M0 não commitada; apenas inspeção de metadados, Git e fonte pública. Nenhum peso ou dado foi baixado.

**Dados:** M001. Checkpoint local SHA-256 `23b4c43f403a4069e3acaa2a11c9a7be971b14e3c363739a6c4d24651c3feee7`.

**Configuração:**

```yaml
seed: 42
network:
  read_only_metadata_search: true
  downloads: false
training: false
embedding_extraction: false
```

**Comandos:**

```bash
sha256sum models/scGPT_heart/best_model.pt
git log --all -- models/scGPT_heart src/approaches/llm/T1/fine_tuning
# Busca somente de metadados públicos pelo SHA/configuração; sem download.
```

**Métricas:**

| Verificação | Resultado | Decisão |
|---|---|---|
| Hash local = objeto LFS publicado | sim | fonte exata identificada |
| Arquitetura | 12 camadas, 8 cabeças, embedding 512 | dimensão conhecida |
| Licença declarada pelo card | MIT | declarada, não basta para liberar |
| Corpus e estágios que geraram o checkpoint | não auditáveis | bloqueio mantido |
| Extração, treino ou download | 0 | dentro da política |

**Artefatos:**

- `docs/M1_PROVENANCE.md`;
- atualização de M001 em `docs/DATA_POLICY.md`;
- nenhum artefato pesado novo.

**Resultado:** O hash coincide com o objeto LFS de `perturblab/scgpt-heart`, que declara MIT e reupload de pesos originalmente obtidos de `bowang-lab/scGPT`. O `args.json` local/remoto cita `cellxgene/.../heart/all_counts`, mas não lista datasets, organismos, estágios, filtros ou exclusão de E10.5–E13.5. Essa ausência impede uso oficial segundo a política.

**Limitações:** identidade de arquivo não prova corpus de pré-treino/fine-tuning. Nenhuma fonte local ou pública encontrada vincula o checkpoint a um manifesto temporal auditável.

**Decisão:** `BLOQUEADO_POR_PROVENIENCIA`; não iniciar decoder, embeddings, agrupamento ou Transformer.

**Próximo experimento:** aguardar um manifesto verificável do corpus ou indicação explícita de outro checkpoint previamente registrado e auditável.

### E005 — Triagem de candidato oficial scGPT

**Data:** 2026-09-23

**Marco:** M1

**Responsável/agente:** Codex

**Hipótese:** O checkpoint oficial `whole-human` pode evitar o problema de proveniência do artefato heart local.

**Critério de decisão definido antes da execução:** registrar o candidato antes de qualquer download; permitir obtenção apenas se houver URL/versionamento/hash do objeto, licença aplicável e evidência suficiente para aplicar a exclusão temporal da competição.

**Código:** inspeção de documentação pública oficial; sem alteração de pipeline, download, treino ou inferência.

**Dados:** M002, registrado nesta execução.

**Configuração:**

```yaml
source: bowang-lab/scGPT official Model Zoo
candidate: whole-human
network: metadata_only
downloaded_weights: false
seed: 42
```

**Métricas:**

| Verificação | Resultado |
|---|---|
| Fonte oficial encontrada | sim |
| Descrição de corpus | 33 milhões de células humanas normais |
| Checkpoint/version/hash específico | não selecionado/não registrado |
| Estágios e manifesto de datasets | ausentes na descrição consultada |
| Download/uso no pipeline | não |

**Resultado:** O Model Zoo oficial recomenda `whole-human` e o descreve como pré-treinado em 33 milhões de células humanas normais. A documentação da construção de corpus cita CellXGene Census e filtros por tecido/doença, mas não fornece um manifesto rastreável do objeto `whole-human` nem resolve como as regras temporais da competição se aplicam a esse corpus.

**Limitações:** compatibilidade biológica humano→camundongo não foi avaliada; ausência de um hash do objeto impede reproduzir a fonte; a fonte não é liberada apenas por ser oficial.

**Decisão:** M002 foi registrado como `PENDENTE`; não baixar ou usar pesos até completar a auditoria e decisão responsável.

**Próximo experimento:** obter somente metadados que fixem objeto/version/hash e o manifesto do corpus, ou receber uma fonte alternativa já aprovada.

### E006 — Autorização de uso exploratório do M001

**Data:** 2026-09-24

**Marco:** M1

**Responsável/agente:** Codex, autorização do responsável do projeto.

**Hipótese:** O checkpoint M001 pode ser usado para o experimento acadêmico desde que o pipeline o marque como não submetível.

**Critério de decisão definido antes da execução:** manter o bloqueio para submissão oficial e permitir somente extração, decoder e avaliação locais do M1.

**Dados:** M001, com hash e fonte registrados; nenhum download adicional.

**Resultado:** autorização explícita recebida em 2026-09-24. Situação atualizada para `USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO`.

**Decisão:** prosseguir com M1, preservando a proibição de uso oficial em configuração, documentação e artefatos.

### E007 — Round-trip linear congelado

**Data:** 2026-09-24

**Marco:** M1

**Hipótese:** Um decoder linear não negativo sobre embeddings scGPT congelados melhora reconstrução sobre repetir o perfil médio sem eliminar diversidade.

**Critério:** MSE de validação menor que o perfil médio, saída de 32.285 genes finita/não negativa e variância/covariância populacionais reportadas sem colapso.

**Configuração:** seed 42; 64 células por estágio; 102/26 treino/validação; 256 genes de entrada, `<cls>`, mapeamento exato/caixa alta; encoder 512 congelado; decoder linear + Softplus; 8 épocas; lote de embedding 8; lote de decoder 16; AdamW `1e-3`; 4 threads CPU.

**Métricas:** MSE `0,061462` vs. média `0,064709`; pseudobulk Pearson `0,97737`; razão de variância `0,06750`; Spearman de variância `0,71896`; Pearson de covariância `0,88253`.

**Resultado/decisão:** saída válida estruturalmente e MSE 5,0% melhor, mas diversidade colapsada. Rejeitar como decoder final; não iniciar M2.

### E008 — Decoder não linear com perda de variância

**Data:** 2026-09-24

**Marco:** M1

**Hipótese:** uma camada oculta de 512 e peso de variância 5 aumentam diversidade sem regressão material.

**Configuração:** idêntica a E007, exceto MLP 512 e `variance_weight=5`.

**Métricas:** MSE `0,076016`; razão de variância `0,26602`; pseudobulk Pearson `0,95281`; Pearson de covariância `0,63033`.

**Resultado/decisão:** diversidade melhora, mas perde para o perfil médio e reduz covariância. Rejeitar.

### E009 — Decoder linear com perda de variância moderada

**Data:** 2026-09-24

**Marco:** M1

**Hipótese:** `variance_weight=1` preserva o ganho de MSE de E007 e recupera diversidade.

**Configuração:** idêntica a E007, exceto `variance_weight=1`.

**Métricas:** MSE `0,062365` vs. média `0,064709`; razão de variância `0,12832`; pseudobulk Pearson `0,97681`; Spearman de variância `0,71985`; Pearson de covariância `0,82723`.

**Resultado/decisão:** melhora MSE 3,6% e dobra a variância de E007, mas a variância continua muito abaixo do alvo. Rejeitar como decoder final e permanecer em M1.

### E010 — Maior cobertura de genes de entrada

**Data:** 2026-09-24

**Marco:** M1

**Hipótese:** aumentar os genes de entrada de 256 para 512 recupera diversidade sem necessidade de alterar o decoder linear.

**Configuração:** idêntica a E007, exceto `input_genes=512`.

**Métricas:** MSE `0,061478` vs. média `0,064709`; razão de variância `0,06093`; pseudobulk Pearson `0,97739`; Spearman de variância `0,71945`; Pearson de covariância `0,87559`.

**Resultado/decisão:** MSE praticamente igual a E007, mas variância ainda menor. Rejeitar; a próxima intervenção M1 deve atuar no objetivo/decoder populacional, não somente em cobertura de entrada.

### E011 — Perda de covariância populacional normalizada

**Data:** 2026-09-24

**Marco:** M1

**Responsável/agente:** Codex

**Hipótese:** Acrescentar ao decoder linear de E007 uma perda de covariância, calculada em 128 genes de maior variância do treino, aumenta conjuntamente a variância (diagonal) e a covariação (fora da diagonal), preservando MSE inferior ao baseline de perfil médio.

**Critério de decisão definido antes da execução:** manter o mesmo split e orçamento de E007; aceitar somente se a saída cumprir o contrato, mantiver MSE abaixo de `0,064709` e apresentar ganho de diversidade/covariação suficiente para afastar o colapso observado. Caso a razão de variância permanecesse próxima de zero, rejeitar e não iniciar M2.

**Código:** árvore de trabalho sem commit (com alterações M0/M1 pré-existentes preservadas). `src/scripts/run_scgpt_roundtrip.py` recebeu apenas `normalized_covariance_loss` e os argumentos opt-in de covariância; `tests/test_scgpt_roundtrip.py` ganhou teste unitário da perda. Não houve alteração de encoder, dados, split ou decoder linear.

**Dados:** D001 e D002; M001 com situação `USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO`. Amostra determinística de 64 células por estágio; 102 treino e 26 validação estratificados por estágio. Não houve download, dados externos, ortologia ou uso de API.

**Configuração:**

```yaml
seed: 42
encoder: scGPT heart M001, congelado, CLS 512
decoder: linear 512 -> 32285 + Softplus
input_genes: 256
epochs: 8
embedding_batch_size: 8
decoder_batch_size: 16
learning_rate: 0.001
variance_weight: 0
covariance_weight: 0.1
covariance_genes: 128
validation_fraction: 0.2
torch_threads: 4
gene_token_mapping: exact_or_uppercase_only
```

**Comandos:**

```bash
uv run python src/scripts/run_scgpt_roundtrip.py \
  --e85 data/E85.h5ad --e95 data/E95.h5ad \
  --output-dir /tmp/ve-m1/E011_covariance_w0.1 \
  --cells-per-stage 64 --input-genes 256 --epochs 8 \
  --embedding-batch-size 8 --decoder-batch-size 16 \
  --learning-rate 0.001 --validation-fraction 0.2 \
  --torch-threads 4 --seed 42 \
  --covariance-weight 0.1 --covariance-genes 128
```

**Métricas:**

| Métrica | Perfil médio | E007 | E009 | E010 | E011 |
|---|---:|---:|---:|---:|---:|
| MSE decoder | 0,064709 | 0,061462 | 0,062365 | 0,061478 | 0,061889 |
| Pearson pseudobulk | — | 0,97737 | 0,97681 | 0,97739 | 0,97580 |
| Razão de variância | 0,00000 | 0,06750 | 0,12832 | 0,06093 | 0,07601 |
| Spearman variância gênica | — | 0,71896 | 0,71985 | 0,71945 | 0,71896 |
| Pearson covariância amostrada | — | 0,88253 | 0,82723 | 0,87559 | 0,89252 |

**Artefatos:**

- temporários não versionados: `/tmp/ve-m1/E011_covariance_w0.1/{decoder.pt,validation_roundtrip.npz,config.json,metrics.json}`;
- `config.json`: hashes de genes de entrada e dos 128 genes de covariância, hash do checkpoint e status não submetível;
- execução: aproximadamente 21 s; a única densificação persistente de treino é `102 × 32.285` `float32` (~12,6 MiB), não a matriz integral dos estágios.

**Resultado:** a saída de validação é `26 × 32.285`, `float32`, finita e não negativa. O MSE é 4,36% inferior ao perfil médio, mas a razão de variância chegou apenas a 0,07601: melhora absoluta de 0,00851 sobre E007 e inferior a E009. A covariação amostrada sobe marginalmente para 0,89252, mas isso não reverte o colapso da diversidade.

**Limitações:** a perda usa covariâncias de batches de até 16 células e apenas 128 genes, logo é um estimador ruidoso e não obriga a reconstrução da estrutura populacional completa. A execução continua exploratória e não submetível porque M001 permanece sem proveniência temporal auditável.

**Decisão:** rejeitar E011 como decoder final. Permanecer em M1; M2, gerador e Transformer temporal continuam bloqueados.

**Próximo experimento:** sem executá-lo automaticamente, comparar uma perda de momentos populacionais calculada sobre todo o conjunto de treino (ou acumulada de forma determinística), incluindo variância e covariância, contra E007 no mesmo split e orçamento; definir antes o peso e o limiar de diversidade.

### E012 — Projeção vazada em referência E9.5

**Data:** 2026-09-24

**Marco:** M1

**Hipótese:** Projetar cada saída já decodificada na célula mais próxima da amostra E9.5 restaura parte da diversidade e das relações gene–gene necessárias para seguir exploratoriamente o pipeline.

**Autorização e critério:** o responsável autorizou explicitamente o uso de vazamento neste diagnóstico de disciplina. A referência contém inclusive células de validação; o resultado deve ser rotulado como diagnóstico, nunca generalização, submissão ou qualidade intrínseca do decoder.

**Configuração:** mesma de E011 (seed 42, encoder congelado, decoder linear, perda de covariância peso 0,1), mais `--snap-to-e95`. Após decodificar, a distância L2 completa seleciona uma das 64 células E9.5 amostradas.

**Comando:**

```bash
uv run python src/scripts/run_scgpt_roundtrip.py \
  --e85 data/E85.h5ad --e95 data/E95.h5ad \
  --output-dir /tmp/ve-m1/E012_leaky_e95_snap \
  --cells-per-stage 64 --input-genes 256 --epochs 8 \
  --embedding-batch-size 8 --decoder-batch-size 16 \
  --learning-rate 0.001 --validation-fraction 0.2 \
  --torch-threads 4 --seed 42 --covariance-weight 0.1 \
  --covariance-genes 128 --snap-to-e95
```

**Métricas:** saída pós-projeção: MSE `0,095270`, pseudobulk `0,91757`, razão de variância `0,39456`, Spearman de variância `0,84223`, covariância `0,70964`; selecionou somente 6 das 64 referências. Decoder cru, registrado separadamente: MSE `0,061889`, razão de variância `0,07601`, covariância `0,89252`.

**Resultado/decisão:** aumentou variância, mas piorou MSE contra o perfil médio (`0,064709`) e covariância; não é uma aproximação útil por si só. Preservado apenas como diagnóstico vazado e como motivação para o teto E013.

### E013 — Correção por resíduo-oráculo da validação

**Data:** 2026-09-24

**Marco:** M1

**Hipótese:** Somar `x_validação - decoder(z_validação)` após a decodificação restaura exatamente a matriz conhecida e confirma que o restante do caminho de artefatos e métricas preserva a diversidade quando o resíduo está disponível.

**Autorização e limitação:** experimento explicitamente autorizado pelo responsável como vazamento exploratório de disciplina. A correção depende da expressão-alvo e é matematicamente uma identidade; não mede capacidade preditiva, não pode ser submetida e não é um decoder utilizável em alvo oculto.

**Configuração/comando:** idênticos a E011, trocando o pós-processamento por `--add-validation-residual`; seed 42, encoder congelado e status M001 `USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO` preservados.

```bash
uv run python src/scripts/run_scgpt_roundtrip.py \
  --e85 data/E85.h5ad --e95 data/E95.h5ad \
  --output-dir /tmp/ve-m1/E013_oracle_residual \
  --cells-per-stage 64 --input-genes 256 --epochs 8 \
  --embedding-batch-size 8 --decoder-batch-size 16 \
  --learning-rate 0.001 --validation-fraction 0.2 \
  --torch-threads 4 --seed 42 --covariance-weight 0.1 \
  --covariance-genes 128 --add-validation-residual
```

**Métricas:** MSE `1,78734e-17` vs. perfil médio `0,064709`; pseudobulk `~1`; razão de variância `~1`; Spearman de variância `~1`; Pearson de covariância `1`. A diferença máxima em valor absoluto para o alvo foi `2,38419e-7`, compatível com `float32`. O decoder cru mantém os números E011, registrados no mesmo `metrics.json`.

**Artefatos e contrato:** `/tmp/ve-m1/E012_leaky_e95_snap/` e `/tmp/ve-m1/E013_oracle_residual/` contêm arrays, configuração, métricas e checkpoint temporários, sem versionamento. E013 tem shape `26 × 32.285`, `float32`, finito e não negativo; a ordem de genes continua a da matriz oficial validada no carregamento.

**Decisão:** o critério científico de decoder sem colapso continua não atendido pelos números crus. Contudo, por autorização explícita do responsável, E013 fecha M1 somente como diagnóstico de caminho completo com vazamento e libera M2 para continuação **exploratória**. Não usar E012/E013 para submissão nem alegar generalização.

### E014 — Embeddings e tokens populacionais M2

**Data:** 2026-09-24

**Marco:** M2

**Responsável/agente:** Codex; uso dos dados `_ex` autorizado pelo responsável.

**Hipótese:** O agrupamento conjunto dos CLS scGPT permite um vocabulário compartilhado de estados, e a grade K=50/100/200 quantifica o compromisso entre compactação, estabilidade entre seeds e estados pequenos ou ausentes.

**Critério de decisão definido antes da execução:** todas as 44.084 células devem receber atribuição; proporções por estágio devem somar 1; grupos vazios/minúsculos, dispersão e estabilidade entre seeds devem ser reportados; escolher o menor K que ofereça a melhor estabilidade sem ignorar vazios. Top genes são somente diagnóstico. O resultado permanece exploratório e não submetível.

**Código:** árvore de trabalho sem commit, preservando alterações M0/M1 preexistentes. Foi adicionado `src/scripts/run_m2_population_tokens.py` e seu teste de contrato. Nenhum Transformer foi implementado ou treinado.

**Dados:** D001–D011 e M001. Os `_ex` somam 10.240 células de E6.5–E8.5; com D001/D002, o conjunto tem 44.084 células em dez estágios. Todos os arquivos têm 32.285 genes na ordem `807549...b8`. A série `_ex` é `log1p(CP10k)` confirmada numericamente e não apresenta IDs sobrepostos; ver `DATA_POLICY.md` e `M2_POPULATION_TOKENS.md`.

**Configuração:** seed principal 42; estabilidade 43/44; CLS 512; 256 genes de maior variância conjunta mapeáveis no vocabulário; encoder M001 congelado; batch de embedding 32 em GPU; `MiniBatchKMeans`, `n_init=3`, batch 2.048, máximo 100 iterações; K=50/100/200. Os 256 pseudobulks gênicos por grupo são escores simples/interpretáveis, não programas aprendidos.

**Comandos:**

```bash
uv run pytest tests/test_m2_population_tokens.py tests/test_scgpt_roundtrip.py tests/test_m0_contract.py -q
uv run python src/scripts/run_m2_population_tokens.py \
  --inputs data/E65_ex.h5ad data/E675_ex.h5ad data/E70_ex.h5ad data/E725_ex.h5ad \
  data/E75_ex.h5ad data/E775_ex.h5ad data/E80_ex.h5ad data/E825_ex.h5ad \
  data/E85_ex.h5ad data/E85.h5ad data/E95.h5ad \
  --output-dir /tmp/ve-m2/E014 --k 50 100 200 --seeds 42 43 44 \
  --input-genes 256 --embedding-batch-size 32 --scan-block-size 256 \
  --threads 4 --device auto
```

**Métricas:**

| K | Inércia/célula | ARI médio vs seed 42 | L1 proporção global (faixa) | Vazios estágio×grupo | Pequenos 1–9 | Dispersão latente média |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 4,9980 | 0,4248 | 0,2176–0,2480 | 340/500 | 17 | 0,07940 |
| 100 | 4,4613 | 0,3622 | 0,2800–0,2985 | 701/1.000 | 42 | 0,07549 |
| 200 | 4,0159 | 0,3463 | 0,2708–0,2876 | 1.468/2.000 | 77 | 0,07638 |

O erro máximo da soma de proporções foi `1,19e-7`; todos os 50/100/200 grupos globais foram usados. A comparação pertinente dentro da grade mostra que aumentar K melhora compactação, mas reduz estabilidade e aumenta fragmentação. K=50 é o baseline escolhido, não uma configuração considerada estável em termos absolutos.

**Artefatos:** `/tmp/ve-m2/E014/`: `embeddings.npz` (CLS, estágios, fontes e 256 genes; ~113 MB), `tokens_k{50,100,200}.npz` (rótulos, centros, proporções, médias, dispersões, escores, top genes, máscaras e resíduos), `config.json`, `metrics.json`. Todos são temporários e ignorados pelo Git.

**Falhas registradas:** a primeira leitura fatiava CSC em modo backed e foi interrompida por I/O excessivo; corrigida para carregar uma fonte por vez. O Lloyd K-means integral com `n_init=10` também foi interrompido sem métricas por custo desproporcional; substituído pelo MiniBatchKMeans declarado acima, reutilizando o cache idêntico.

**Resultado:** contrato estrutural atendido e K=50 selecionado como baseline exploratório. Estágios iniciais muito pequenos (E6.5 tem quatro células) tornam muitos vazios esperados; a estabilidade moderada impede interpretar IDs de cluster como estados biológicos rígidos.

**Decisão:** encerrar M2 exploratório com K=50. Não iniciar M3 nem qualquer Transformer sem nova autorização. M001 continua `USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO`.

### E015 — Gerador conhecido por centros e resíduos

**Data:** 2026-09-24

**Marco:** M3

**Hipótese:** Ao reconstruir E9.5 conhecido, amostrar resíduos reais dentro do mesmo `(estágio, grupo)` preserva mais diversidade que repetir o centro de cada grupo, sem piorar materialmente pseudobulk, MMD e covariância.

**Critério de decisão definido antes da execução:** usar K=50 e seed 42; exigir saída `float32`, finita, não negativa e com 32.285 genes; comparar contra centroides sob as mesmas proporções; reportar composição, MMD aproximada, covariância, diversidade e sensibilidade à contagem. Aceitar o gerador somente se resíduos aumentarem diversidade e não regredirem as métricas populacionais principais.

**Código:** árvore sem commit; adicionados `run_m3_known_population.py` e testes. O cache E014 não foi decodificado porque seu conjunto de genes (`d842…`) difere do decoder E011 (`0b94…`). Foi reextraído um cache K=50 compatível com E011, sem treinar encoder/decoder e sem Transformer.

**Dados:** D001–D011, M001 e decoder E011. E9.5 real D002 é apenas alvo conhecido de reconstrução, não previsão temporal. Amostra de avaliação: 512 células determinísticas de D002.

**Configuração:** seed 42; K=50; CLS 512; 256 genes exatos de E011; `MiniBatchKMeans(n_init=3,batch_size=2048,max_iter=100)`; 512 células decodificadas; sensibilidade latente em 128/512/2.048; decoder linear + Softplus E011 congelado. MMD de avaliação usa 256 Random Fourier Features sobre 256 genes variáveis: `MMD² ≈ ||mean(phi(x))-mean(phi(y))||²`, custo linear em células e features, sem kernel O(n²). Não foi usada como loss.

**Comandos:**

```bash
uv run python src/scripts/run_m2_population_tokens.py ... \
  --output-dir /tmp/ve-m3/E015/compatible_m2 --k 50 --seeds 42 43 \
  --selected-genes-from /tmp/ve-m1/E011_covariance_w0.1/decoder.pt
uv run python src/scripts/run_m3_known_population.py \
  --cache /tmp/ve-m3/E015/compatible_m2/embeddings.npz \
  --tokens /tmp/ve-m3/E015/compatible_m2/tokens_k50.npz \
  --decoder /tmp/ve-m1/E011_covariance_w0.1/decoder.pt \
  --target data/E95.h5ad --output-dir /tmp/ve-m3/E015/generation \
  --stage E9.5 --n-cells 512 --sensitivity-cells 128 512 2048 --seed 42
uv run pytest tests/test_m3_known_population.py tests/test_m2_population_tokens.py \
  tests/test_scgpt_roundtrip.py tests/test_m0_contract.py -q
```

**Métricas:**

| Métrica | Centroides | Centros + resíduos | Decisão |
|---|---:|---:|---|
| células de expressão únicas / 512 | 45 | 506 | resíduos preservam diversidade discreta |
| razão de variância | 0,07037 | 0,07715 | melhora pequena; colapso persiste |
| pseudobulk Pearson | 0,98712 | 0,98704 | sem melhora |
| erro relativo de pseudobulk | 0,14982 | 0,15045 | regressão pequena |
| covariância Pearson, 64 genes | 0,96159 | 0,96161 | empate |
| RFF-MMD², 256 genes | 0,08703 | 0,08915 | regressão pequena |

Sensibilidade: L1 de composição foi 0,4310/0,2064/0,1178 para 128/512/2.048 células; razão de variância latente foi 0,9820/0,9790/1,0149. Não houve fallback em nenhuma contagem.

**Artefatos:** `/tmp/ve-m3/E015/compatible_m2/` e `/tmp/ve-m3/E015/generation/`, incluindo cache, tokens/resíduos, configuração, métricas e dois `.h5ad` de 512 × 32.285. Ambos passaram o contrato (`float32`, finitos, não negativos e ordem oficial).

**Resultado:** o gerador por resíduos funciona no latente e evita repetição maciça, mas o decoder E011 elimina quase todo o ganho: a variância final continua em 7,7% do alvo e as métricas populacionais não superam centroides.

**Decisão:** resultado negativo e gate M3 não atendido. Não avançar para dinâmica temporal. A próxima intervenção deve tratar o decoder/compatibilidade de geração, com ablação controlada, antes de qualquer Transformer.

### E016 — Transferência de resíduos de expressão E8.5→E9.5

**Data:** 2026-09-24

**Marco:** M3/M4, extensão exploratória autorizada.

**Responsável/agente:** Codex; autorização explícita do responsável para avançar apesar dos gates M1/M3.

**Hipótese:** Somar resíduos de expressão conhecidos de E8.5, preferencialmente do mesmo grupo K=50, à expressão decodificada de latentes E9.5 reconstruídos recupera diversidade e aproxima a distribuição real de E9.5 melhor que o decoder cru, sem usar qualquer resíduo real de E9.5.

**Critério de decisão definido antes da execução:** usar seed 42 e a tokenização compatível com E011; comparar decoder cru, centroides, decoder + resíduo E8.5 e vizinho E8.5 sob a mesma amostra E9.5; registrar fallbacks de grupos vazios, células únicas, pseudobulk, razão de variância, covariância e RFF-MMD. Considerar a transferência promissora somente se melhorar diversidade e ao menos uma métrica distribucional sem regressão material generalizada. Resultado negativo será registrado, mas não bloqueará E017/E018 pela autorização operacional vigente.

**Código:** árvore de trabalho sem commit, preservando todas as alterações anteriores.

**Dados:** D001 como única fonte de resíduos/vizinhos; D002 somente como pseudoalvo conhecido de avaliação; M001 e decoder E011 exploratórios. Nenhum E10.5 real e nenhuma fonte nova.

**Configuração:** K=50, seed 42, 512 células por estágio, decoder E011 congelado, lote 32, RFF-MMD com 256 features sobre os 256 genes mais variáveis do alvo.

**Comando:**

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python src/scripts/run_e016_expression_residual.py \
  --cache /tmp/ve-m3/E015/compatible_m2/embeddings.npz \
  --tokens /tmp/ve-m3/E015/compatible_m2/tokens_k50.npz \
  --decoder /tmp/ve-m1/E011_covariance_w0.1/decoder.pt \
  --e85 data/E85.h5ad --e95 data/E95.h5ad \
  --output-dir /tmp/ve-m4/E016 --n-cells 512 --seed 42 --threads 4
```

**Métricas:**

| Variante | Pseudobulk Pearson | Erro pseudobulk | Razão de variância | Covariância Pearson | RFF-MMD² | Únicas/512 |
|---|---:|---:|---:|---:|---:|---:|
| decoder cru | 0,98716 | 0,14859 | 0,07705 | 0,95762 | 0,08190 | 512 |
| decoder + resíduo E8.5 | 0,99357 | 0,10519 | 0,95173 | 0,96302 | 0,01056 | 512 |
| centroides | 0,98719 | 0,14825 | 0,06911 | 0,95746 | 0,08459 | 88 |
| vizinho E8.5 | 0,99298 | 0,11011 | 0,94766 | 0,96000 | 0,01151 | 255 |

**Artefatos:** `/tmp/ve-m4/E016/` contém quatro `.h5ad` 512 × 32.285, `config.json`, `metrics.json` e `contracts.json`. Todos os arquivos passaram o contrato de genes, ordem, `float32`, finitude e não negatividade.

**Resultado:** a transferência usou somente resíduos `x_E8.5 - decoder(z_E8.5)`. Um grupo de E9.5 estava ausente na amostra E8.5 e causou fallback em 1/512 célula para o grupo não vazio de centro global mais próximo. A correção superou decoder cru, centroides e vizinho E8.5 nas métricas principais e preservou 512 expressões únicas.

**Limitações:** pseudo-holdout conhecido, amostra pequena e vocabulário K=50 ajustado conjuntamente incluindo E9.5; portanto o resultado valida o mecanismo, não generalização temporal. O resíduo corrige principalmente limitações do decoder e pode transportar estado de E8.5.

**Decisão:** adotar `decoder + resíduo de expressão do último estágio conhecido` como variante principal exploratória; manter decoder cru e vizinho como comparadores. Prosseguir para E017 por autorização explícita.

### E017 — Baselines temporais em tokens

**Data:** 2026-09-24

**Marco:** M4/M5.

**Hipótese:** extrapolação linear entre E8.0/E8.25 aproxima os tokens E8.5 melhor que copiar E8.25 em pelo menos uma família de saídas, mas pode gerar centros/dispensões instáveis; ambos definem referências obrigatórias para E018.

**Critério de decisão definido antes da execução:** avaliar copy-last e extrapolação linear contra os mesmos tokens K=50 de E8.5; proporções devem ser finitas, não negativas e somar 1, dispersões devem ser não negativas; reportar MSE de proporções, centros e dispersões com máscara do alvo, além do erro agregado. Não selecionar configuração por informação de E8.5 além da avaliação. O vocabulário conjunto inclui o holdout e será registrado como vazamento estrutural.

**Configuração:** K=50; seed 42; D009/E8.0 e D010/E8.25 como entradas; D011/E8.5 como alvo; fator temporal linear 1,0. Vocabulário K=50 conjunto de E014/E015.

**Comando:**

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python src/scripts/run_e017_temporal_baselines.py \
  --cache /tmp/ve-m3/E015/compatible_m2/embeddings.npz \
  --tokens /tmp/ve-m3/E015/compatible_m2/tokens_k50.npz \
  --output-dir /tmp/ve-m5/E017 --seed 42
```

**Métricas:**

| Método | MSE proporção | L1 proporção | MSE centros | MSE dispersões | MSE agregado |
|---|---:|---:|---:|---:|---:|
| copy-last | 0,001592 | 0,57423 | 0,001770 | 0,000359 | 0,003721 |
| extrapolação linear | 0,0000687 | 0,10728 | 0,162743 | 0,001075 | 0,163887 |

**Falha operacional registrada:** a primeira execução resumiu D011 e D001 juntos porque ambos têm rótulo `E8.5`; produziu métricas inválidas (agregado 0,848/0,928) e foi sobrescrita antes de congelar E017. O script foi corrigido para resumir por `sources`, e a execução válida usa exclusivamente D009–D011. Nenhum embedding foi reextraído.

**Resultado:** a extrapolação linear prevê composição muito melhor, mas extrapola centros de grupos raros/vazios de forma instável e perde amplamente no erro agregado. Copy-last é o baseline agregado a superar.

**Limitações:** somente seis grupos estão presentes no pequeno D011; os centros são avaliados apenas sob máscara do alvo. O K-means foi ajustado conjuntamente incluindo o holdout e os estágios oficiais, um vazamento estrutural explícito que impede alegação de generalização estrita.

**Decisão:** manter ambos como baselines; reservar E018.

### E018 — Transformer temporal mínimo

**Data:** 2026-09-24

**Marco:** M5.

**Hipótese:** Um Transformer numérico pequeno, treinado nas janelas conhecidas anteriores a E8.5, reduz o erro agregado de tokens no holdout D011/E8.5 em relação ao copy-last sem produzir proporções ou dispersões inválidas.

**Critério de decisão definido antes da execução:** `model_dim=128`, duas camadas, quatro cabeças, dropout pequeno, seed 42 e treino curto configurável; entradas de dois estágios e tempo alvo; loss simples de proporção, centro e dispersão mascaradas. Não usar D011 na normalização, treino, early stopping ou seleção. Comparar uma MLP temporal simples sob o mesmo split se o custo for baixo. Aceitar validade operacional se todas as saídas passarem os contratos e a execução for reproduzível; superioridade científica exige erro agregado menor que copy-last. Resultado negativo não interrompe a extrapolação E10.5 autorizada.

**Configuração:** seed 42; CLS 512/K=50; entradas de duas etapas; `model_dim=128`; duas camadas; quatro cabeças; dropout 0,05; AdamW `1e-3`; 250 épocas; seis janelas de treino com alvos E7.0–E8.25; D011/E8.5 exclusivamente holdout; MLP 128 sob o mesmo split. Normalização escalar calculada somente nos estados de treino.

**Comando:**

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python src/scripts/run_e018_population_transformer.py \
  --cache /tmp/ve-m3/E015/compatible_m2/embeddings.npz \
  --tokens /tmp/ve-m3/E015/compatible_m2/tokens_k50.npz \
  --output-dir /tmp/ve-m5/E018 --epochs 250 --model-dim 128 \
  --layers 2 --heads 4 --dropout 0.05 --learning-rate 0.001 --seed 42 --threads 4
```

**Métricas:**

| Método | MSE proporção | L1 proporção | MSE centros | MSE dispersões | MSE agregado |
|---|---:|---:|---:|---:|---:|
| copy-last E017 | 0,001592 | 0,57423 | 0,001770 | 0,000359 | **0,003721** |
| linear E017 | 0,0000687 | **0,10728** | 0,162743 | 0,001075 | 0,163887 |
| MLP | 0,004693 | 0,94933 | 0,004472 | **0,000248** | 0,009413 |
| Transformer | 0,005810 | 1,06803 | 0,013487 | 0,000461 | 0,019757 |

Loss final de treino: Transformer `0,006519`; MLP `0,012667`. O Transformer atribuiu massa >1e-6 a 49 grupos, contra seis observados; a MLP atribuiu a oito.

**Artefatos:** `/tmp/ve-m5/E018/{transformer.pt,mlp.pt,predictions.npz,config.json,metrics.json,training_history.json}`.

**Resultado:** critérios operacionais atendidos (treino/inferência reprodutíveis; shapes válidos; proporções finitas/não negativas somando 1; dispersões positivas). A hipótese de superioridade foi rejeitada: copy-last vence no erro agregado, e a MLP sem atenção vence o Transformer.

**Limitações:** seis exemplos de treino, estágios iniciais extremamente pequenos, forte mudança de composição e vocabulário K=50 ajustado conjuntamente incluindo holdout/estágios oficiais. O resultado não estima generalização para E10.5.

**Decisão:** manter copy-last como recomendação de dinâmica segundo holdout; usar o Transformer somente para concluir a extrapolação exploratória autorizada e preservar as três variantes de expressão.

### E019 — Predição E10.5 exploratória ponta a ponta

**Data:** 2026-09-24

**Marco:** M5/M7 exploratório.

**Hipótese:** Após reajuste nas janelas conhecidas D003–D011, o Transformer operacional pode extrapolar tokens a E10.5 a partir dos estados oficiais E8.5/E9.5; resíduos de expressão de E9.5 devem preservar mais diversidade que o decoder cru, sem consultar E10.5 real.

**Critério de decisão definido antes da execução:** gerar 512 células com seed 42; proporções por softmax e dispersões por softplus; gerar latentes por centro/dispensão e resíduos latentes E9.5; comparar decoder cru, decoder + resíduo de expressão E9.5 e vizinho E9.5 usando apenas diagnósticos intrínsecos e distância a E9.5, nunca como score de E10.5. Exigir 32.285 genes na ordem oficial, `float32`, finitude e não negatividade. Registrar fallbacks. M001 mantém toda saída como não submetível.

**Configuração:** seed 42; sete janelas com alvos E7.0–E8.5; 250 épocas; Transformer 128/2 camadas/4 cabeças/dropout 0,05; AdamW `1e-3`; entradas finais D001/E8.5 e D002/E9.5; alvo temporal 10.5; 512 células; escala de resíduo latente limitada a `[0,3]`. Nenhum dado E10.5 foi lido.

**Comando:**

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python src/scripts/run_e019_predict_e10_5.py \
  --cache /tmp/ve-m3/E015/compatible_m2/embeddings.npz \
  --tokens /tmp/ve-m3/E015/compatible_m2/tokens_k50.npz \
  --decoder /tmp/ve-m1/E011_covariance_w0.1/decoder.pt --e95 data/E95.h5ad \
  --output-dir /tmp/ve-m7/E019 --n-cells 512 --epochs 250 \
  --model-dim 128 --layers 2 --heads 4 --dropout 0.05 \
  --learning-rate 0.001 --seed 42 --threads 4
```

**Métricas diagnósticas contra amostra E9.5 (não são métricas do alvo E10.5):**

| Variante | Pseudobulk Pearson | Erro pseudobulk | Razão de variância | Covariância Pearson | RFF-MMD² | Únicas/512 |
|---|---:|---:|---:|---:|---:|---:|
| decoder cru | 0,94952 | 0,30487 | 0,04075 | 0,92011 | 0,28924 | 473 |
| decoder + resíduo E9.5 | **0,96292** | **0,27273** | **0,88172** | 0,59410 | 0,11572 | 473 |
| vizinho E9.5 | 0,94719 | 0,32206 | 0,68891 | 0,69983 | **0,10630** | 83 |

As proporções previstas somaram exatamente 1; 34 grupos receberam ao menos uma célula e não houve fallback. A correção residual preservou mais diversidade, mas reduziu a correlação de covariância em relação ao decoder cru.

**Artefatos:** `/tmp/ve-m7/E019/` contém `predicted_tokens.npz`, `transformer_final.pt`, configuração, histórico, métricas, contratos e:

- `e10_5_decoder_raw_512.h5ad`;
- `e10_5_expression_residual_e95_512.h5ad` (variante principal);
- `e10_5_nearest_e95_512.h5ad` (baseline agressivo).

Validação independente confirmou em cada arquivo shape `512 × 32.285`, genes na ordem oficial, `.X float32`, valores finitos e não negativos. Máximos: 5,861/6,795/6,224, compatíveis com a faixa log-normalizada observada. Os arquivos são densos e temporários.

**Validação de código:** a suíte focada terminou com `20 passed`; `git diff --check` e compilação dos cinco módulos novos passaram. A suíte integral continua interrompida na coleta pelos dois problemas históricos já registrados: ausência de `src.approaches.mouse_geneformer` e `src.scripts.subset_heart`. Nenhuma falha nova apareceu na suíte focada.

**Resultado:** pipeline ponta a ponta concluído sem correspondência célula-a-célula e sem acesso ao alvo oculto. A variante residual é a recomendação exploratória por recuperar diversidade e evitar a memorização mais forte do vizinho, mas não há evidência de qualidade em E10.5.

**Limitações:** M001 não tem proveniência temporal auditável e torna os resultados não submetíveis; Transformer perde para copy-last no holdout; apenas sete janelas de treino, mudança de domínio entre atlas externo e dados oficiais, K=50 ajustado incluindo holdout, decoder colapsado e 512 células possivelmente abaixo do limite oficial. Métricas E019 medem distância a E9.5, não verdade E10.5.

**Decisão:** concluir o objetivo acadêmico exploratório. Não selecionar uma submissão por essas métricas. Antes de submissão, resolver proveniência, reconfirmar regras/contagem, executar validador oficial e preferir dinâmica respaldada pelo holdout (copy-last/MLP) ou obter supervisão temporal permitida adicional.

### E020 — Transformer v2 com DE/ranking e RFF-MMD

**Data:** 2026-09-24. **Marco:** M5/M7. **Seed:** 42.

**Hipótese e critério:** losses diferenciáveis alinhadas às métricas e seleção por ganho relativo ao copy-last são operacionalmente treináveis. O smoke deveria completar geração, produzir gradientes finitos e três `.h5ad` válidos; não mede qualidade científica.

**Configuração smoke:** 24 células sintéticas, 32 genes selecionados (override barato do valor nominal 512), latente 8, K=5, 16 RFF, `tau=0,1`, top-k 8, bloco 8, duas épocas. Pesos DE/MMD 0,1. A configuração de servidor usa K=50, 2.048 genes, bloco 128, 512 RFF e as quatro arquiteturas/losses registradas em `server.yaml`.

**Comando:** `UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_population_transformer_v2 --config configs/population_transformer/smoke.yaml --smoke-test`.

**Resultado:** loss final `0,4134095` (`L_DE=2,7648654`, `L_MMD=1,3692300`); gradientes finitos. As variantes decoder cru, residual e vizinho passaram contratos `24×32`, `float32`, finitos e não negativos. Artefatos em `/tmp/ve-e020/smoke/`. Testes focados: `16 passed`, uma advertência conhecida do Transformer PyTorch.

Validação barata adicional nos dados reais: com seed 42, D001 foi separado em 14.287 treino + 2.500 reserva (zero sobreposição, 18 rótulos representados) e D002 em 14.557 + 2.500 (zero sobreposição, 21 rótulos). `celltype` foi lido somente para estratificação; nenhuma expressão foi densificada e o rótulo não é entrada do modelo. Um smoke real scGPT→decoder com 8 células/estágio, 32 genes e uma época completou em ~10 s (`MSE=0,22265`, razão de variância `0,05330`); é apenas teste de caminho, não configuração científica.

Um segundo smoke de integração criou artefatos equivalentes a quatro estágios externos + D001/D002, executou as fases `search`, `retrain` e `generate`, preservou checkpoints e produziu as três variantes com exatamente 2.500 células. Para manter custo baixo usou K=3, latente 4, 8 genes, quatro células por loss e uma época; os contratos das três saídas `2500×8` passaram. O score muito negativo da fixture (`-392001,95`) não possui interpretação científica: copy-last tinha erro de proporção exatamente zero e o denominador seguiu o epsilon declarado.

Na primeira execução com os caches reais, `numpy.multinomial` rejeitou proporções K=50 cuja soma em `float32` diferia de 1 após conversão estrita para `float64` (`sum(pvals[:-1]) > 1`). A tentativa é inválida e não gerou run selecionável. A amostragem foi corrigida para normalizar em `float64`, colocar a maior massa na última posição e defini-la como resto exato. O teste de regressão passou. Em seguida, uma busca real reduzida sobre os 44.084 embeddings (uma arquitetura 32/1/4, uma época, 32 genes, oito células por loss) completou oito runs sem erro; o melhor smoke foi estrutural com score relativo `-109,6075`. Esse score reduzido não é resultado científico e não substitui a configuração do servidor.

**Limitações/decisão:** a integração completa está validada apenas em fixture reduzida e não compara qualidade científica. O pipeline de servidor está implementado em quatro fases, mas a busca pesada, o retreino e os três arquivos oficiais `2500×32285` ainda não foram executados. Essa execução pendente deve fornecer as métricas científicas e pode produzir resultado negativo contra copy-last.

### E021 — Transformer residual com geração ancorada

**Data:** 2026-09-24. **Marco:** M5/M7. **Seed:** 42.

**Hipótese:** usar células reais do último estágio como âncoras e aplicar mudanças de baixa dimensão previstas pelo Transformer preservará esparsidade, diversidade e covariação melhor que decodificar embeddings scGPT para os 32.285 genes.

**Critério de decisão definido antes da execução científica:** avaliar primeiro E8.5→E9.5 conhecido; comparar a população final, e não o decoder intermediário, contra `copy_last` com a mesma contagem e seeds; exigir ao menos 95% de células únicas, monitorar zeros/variância/covariação e calcular DE, direção, MMD e variograma nos 32.285 genes. `copy_last` é somente controle e não candidato de saída do método.

**Código/configuração inicial:** `anchored.py` adiciona programas assinados por `TruncatedSVD`, aplicação de deslocamento por grupo sobre expressão real, preservação de zeros sem suporte explícito e amostragem que esgota cada pool antes de repetir. O scGPT permanece somente como encoder neste ciclo controlado. Mouse Geneformer foi autorizado pelo responsável como ablação posterior, mas não foi usado nem auditado neste experimento.

O retreino E020 também foi corrigido: a busca continua escolhendo a época por validação temporal real; o retreino com todos os dados executa exatamente esse orçamento e salva a última época com `validation_score=null`, em vez de usar o índice da época como score crescente.

**Comandos:**

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run pytest \
  tests/test_population_pipeline_v2.py \
  tests/test_population_transformer_v2.py \
  tests/test_population_temporal.py -q
uv run python -m compileall -q \
  src/approaches/population_transformer \
  src/scripts/run_population_transformer_v2.py
git diff --check
```

**Implementação final:** a cabeça de programas foi integrada ao Transformer; scores dos dois estágios entram nos tokens e o modelo prevê deltas normalizados por programa. A geração aplica deltas sobre células reais do último estágio, preserva zeros sem suporte, seleciona âncoras sem reposição e usa shrinkage residual independente para expressão e composição. O decoder scGPT saiu do caminho principal; M001 permanece apenas como encoder.

**Pseudo-holdout confirmado:** E8.25 externo como referência e E8.5 externo como alvo; 1.024 células únicas; todos os 32.285 genes; programas e avaliação seed 42; 32 programas; `α_expr=0,25`; `α_prop=0,25`.

| Métrica `veckit` | `copy_last` | E021 | Variação |
|---|---:|---:|---:|
| `de_score` | 0,0000 | 0,0917 | +0,0917 |
| `de_direction` | 0,0000 | 0,1261 | +0,1261 |
| `mmd_u` | 0,06525 | 0,06169 | -0,00356 |
| `variogram` | 0,001260 | 0,001256 | -0,000004 |
| `pb_rel_err` | 0,1063 | 0,1049 | -0,0014 |
| `composition_JSD` | 0,0071 | 0,0062 | -0,0009 |
| `pseudobulk_pearson` | 0,9939 | 0,9941 | +0,0002 |

Uma população de 512 células com avaliação fixada foi repetida variando somente a inicialização do Transformer. Seeds 43/44 mantiveram ganho de direção, mas não repetiram a melhora simultânea de MMD/variograma; a estabilidade é, portanto, parcial. Um ensemble por mistura preservou DE/direção/MMD, mas piorou variograma e não foi escolhido.

**Geração E10.5:** o modelo final foi retreinado por três épocas — orçamento escolhido pelo melhor checkpoint do pseudo-holdout — usando todas as transições externas permitidas e E8.5→E9.5 oficial. Nenhum dado E10.5 foi lido. O artefato `outputs/population_transformer/E021/final_seed42/e10_5_anchored_2500.h5ad` tem 2.500 × 32.285, `float32`, valores finitos/não negativos, 2.500 células únicas, nenhuma âncora E9.5 repetida e fração de zeros 0,87722.

**Comandos adicionais:**

```bash
UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e021_anchored \
  --output-dir outputs/population_transformer/E021/confirm_seed42_n1024 \
  --n-cells 1024 --n-programs 32 --epochs 250 --patience 25 \
  --seed 42 --program-seed 42 --evaluation-seed 42 --device auto \
  --scales 0.25 1 --proportion-scale 0.25

UV_CACHE_DIR=/tmp/ve-uv-cache uv run python -m src.scripts.run_e021_generate \
  --output-dir outputs/population_transformer/E021/final_seed42 \
  --n-cells 2500 --epochs 3 --seed 42 --device auto
```

O `veckit --task T1` foi executado separadamente para `copy_last`, `anchored_0.25` e `anchored_1`, sempre com o mesmo alvo, referência, contagem, genes e seed.

**Validação de código:** `16 passed`; compilação e `git diff --check` passaram.

**Limitações:** superioridade conjunta foi confirmada somente na seed 42. A quantidade de transições continua pequena, o pseudo-holdout é de domínio externo e a diferença de variograma é muito pequena. A candidata E10.5 é estruturalmente válida, mas não possui avaliação possível contra o alvo oculto.

**Decisão:** concluir E021 como resultado parcialmente positivo e preservar a candidata seed 42. Não trocar para Mouse Geneformer ainda; a próxima intervenção deve reduzir variância entre inicializações mantendo o protocolo de avaliação fixo.

**Resultado no leaderboard informado pelo responsável:**

| Métrica | E020 anterior | E021 | Delta | `copy_last` |
|---|---:|---:|---:|---:|
| `de_score` | 37,9 | 38,8 | +0,9 | 50,0 |
| `de_direction` | 46,2 | 51,4 | +5,2 | 50,0 |
| `mmd_u` | 38,8 | 48,4 | +9,6 | 50,0 |
| `variogram` | 42,7 | 48,5 | +5,8 | 50,0 |
| ponderado oficial | 41,21 | 46,77 | +5,57 | 50,00 |

O resultado confirma que ancoragem em células reais resolveu a maior parte da perda distribucional e de covariação. `de_direction` superou o baseline, enquanto `de_score` permanece 11,2 pontos abaixo e passa a ser o gargalo dominante. A decisão E021 é atualizada para **resultado positivo no leaderboard**, ainda sem superioridade agregada sobre `copy_last`.

### E022 — Loss DE assinada em múltiplos cortes

**Data:** 2026-09-24. **Marco:** M5/M7. **Seed:** 42.

**Hipótese:** uma loss diferenciável separada para genes positivos e negativos, avaliada simultaneamente nos cortes top-50, 100, 250, 500 e 1.000, melhora `de_score` sem regredir MMD e variograma.

**Implementação:** para cada corte, a loss cria objetivos binários balanceados one-vs-all para os maiores deltas positivos e negativos. O delta predito é calculado no pseudobulk final diferenciável, incluindo proporções previstas, expressão-base por grupo e deltas reconstruídos pelos programas. Os cortes são definidos pelo alvo conhecido e destacados do gradiente.

**Configuração:** mesmo pseudo-holdout fixo E021 de 512 células e 32.285 genes; 32 programas; `α_expr=α_prop=0,25`; seeds de programa/avaliação 42; pesos da nova loss `{0,05, 0,1, 0,2}`; temperatura 0,1. O scorer foi executado contra exatamente o mesmo alvo e referência do E021.

| Configuração | `de_score` | `de_direction` | `mmd_u` | `variogram` |
|---|---:|---:|---:|---:|
| E021 sem loss multicut | 0,0476 | 0,1867 | 0,07179 | 0,001332 |
| peso 0,05 | -0,0095 | 0,1481 | 0,07719 | 0,001403 |
| peso 0,10 | -0,0095 | 0,1449 | 0,07822 | 0,001439 |
| peso 0,20 | 0,0190 | 0,1642 | 0,07804 | 0,001407 |

**Validação de código:** o novo teste confirma loss menor para ranking up/down correto que para ranking invertido e gradientes finitos. Suíte focada: `17 passed`; compilação e `git diff --check` passaram.

**Resultado:** nenhum peso supera E021 em recuperação DE; MMD e variograma também regridem. A hipótese foi rejeitada. O provável motivo é incompatibilidade entre uma classificação binária rígida nos 32.285 genes e o subespaço de somente 32 programas, além de competição com as losses estruturais.

**Decisão:** preservar a implementação como ablação opt-in (`--de-weight`, default zero), não retreinar nem submeter nova candidata e manter E021 seed 42 como melhor método.

### E023 — Base híbrida temporal/DE

**Hipótese pré-registrada:** acrescentar direções temporais enriquecidas para sinais positivos/negativos estáveis amplia a capacidade DE sem destruir a distribuição ancorada.

**Protocolo:** E021 fixo, D010 referência e D011 alvo, 32.285 genes, 512 células primeiro; program_seed=evaluation_seed=42; Transformer seed=42 (43/44 somente após gates); escalas expressão/composição 0,25; loss multicut desligada. Comparar 32 gerais, 32+16 DE, 32+32 DE e 64 gerais. Mesma arquitetura 128/2/4, dropout 0,05, AdamW lr=0,001, weight_decay=0,0001, até 250 épocas, patience=25. Seleção de época idêntica a E021.

**Construção:** SVD geral preservada; SVD residual sobre deltas por grupo, pseudobulk e partes positivas/negativas ponderadas por estabilidade de sinal entre transições de treino. Ortogonalização contra a base geral; D011 não participa da construção. Cache scGPT/K50 herdado inclui holdout: limitação estrutural preservada e explicitada.

**Gates definidos antes da execução:** DE estritamente maior que E021; direção pelo menos 95% do E021; MMD/variograma no máximo 102% do E021; células únicas >=95%; zeros a até 2 pontos percentuais do alvo. Confirmação em 1.024 apenas para aprovadas; mesmas referências e alvos, seeds fixos. Não gerar E10.5 antes da confirmação/estabilidade. Parar se nenhuma configuração passar ou após três intervenções negativas consecutivas.

**Código/dados:** árvore não commitada herdada, alterações pequenas opt-in; D003–D010 treino, D011 avaliação/seleção de época, M001 permitido por decisão registrada. Nenhuma leitura E10.5. E021/E022 preservados. GPU detectada: RTX4050 6 GiB; RAM 27 GiB.

**Artefatos:** `outputs/population_transformer/E023/<configuração>/`. Resultados e comandos serão acrescentados após execução.

**Resultados E023 — mesma avaliação 512 células:**

| Configuração | DE ↑ | direção ↑ | MMD ↓ | variograma ↓ | pb_rel_err ↓ | JSD ↓ | PB Pearson ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| copy_last | 0,0000 | 0,0000 | 0,07805 | 0,001375 | 0,1082 | 0,0075 | 0,9937 |
| E021 32 gerais | 0,0476 | 0,1867 | 0,07179 | 0,001332 | 0,1061 | 0,0052 | 0,9940 |
| E023 32+16 | 0,0286 | 0,1593 | 0,07484 | 0,001397 | 0,1079 | 0,0047 | 0,9938 |
| E023 32+32 | 0,0095 | 0,0985 | 0,08337 | 0,001480 | 0,1123 | 0,0099 | 0,9933 |
| 64 gerais | inválido | — | — | — | — | — | — |

32+16 piorou MMD 4,25% e variograma 4,88%; 32+32 piorou 16,13% e 11,11%. Ambas falharam também DE e direção. Todas as populações válidas têm 512/512 células únicas; zeros: 90,586% (32+16), 90,716% (32+32), alvo 90,858%. Contrato 512×32.285, genes/ordem, float32, finitude e não negatividade passaram; alvo e referência idênticos aos E021 confirmados por comparação dos arrays. Erro máximo de ortogonalidade ~1,07e-6. Épocas escolhidas (índice zero): 2 e 11.

**Falha registrada:** 64 gerais pediu mais componentes do que as 37 linhas observadas permitem; falhou explicitamente antes do treino. Registro `general64_seed42_n512/failure.json`. Não houve OOM, leitura de E10.5 ou transferência para servidor.

**Validação e artefatos:** 18 testes focados passaram; compileall e diff check passaram. Auditoria, hashes/tamanhos e configurações em `outputs/population_transformer/E023/audit.json`; hashes de código em `code_hashes.json`. Comandos completos, método e contribuição Ollama descartada em `docs/E023_HYBRID_PROGRAMS.md`. O baseline 32 foi retreinado para verificar ausência de alteração involuntária.

**Decisão:** rejeitar ambas as bases híbridas testadas. Encerrar pela condição explícita de nenhuma configuração passar gates. Não executar 1.024 células, seeds 43/44, ensemble, loss multicut nem gerar E10.5. Não interpretar ausência de melhora como prova de que qualquer programa híbrido falhará: base, previsibilidade e loss normalizada por dimensão continuam acopladas. Não houve ganho DE que justificasse o ramo de shrinkage. E021 segue melhor candidata; copy_last permanece controle.

**Próximo passo recomendado:** separar capacidade de representação e previsibilidade temporal em validações internas somente de treino antes de propor E024; eventual Mouse Geneformer requer auditoria de corpus/licença/hash. Nenhum encoder foi trocado neste ciclo.

### E024 — Correção simples por tendência gênica recente

**Autorização:** responsável pediu nova busca após parada E023, priorizando simplicidade e Ollama para tarefas mecânicas.

**Hipótese pré-registrada:** a tendência de pseudobulk entre os dois últimos estágios conhecidos contém informação gene-específica perdida pelos 32 programas. Somar uma pequena fração dessa tendência à população E021 pode melhorar DE sem novo encoder, loss ou arquitetura. Preservar zeros e não parear células.

**Configuração inicial:** correção `max(x_E021 + w*(pb_E8.25-pb_E8.0)*I(x_E021>0),0)`, w={0,0625;0,125;0,25}. Equivale a expression_scale=0,25 vezes peso de tendência {0,25;0,5;1}. Médias completas apenas D009/D010; referência/alvo fixos E021, 512 células, seeds de programa/modelo/avaliação=42. Não usar D011 na construção. E021 continua núcleo temporal e composição. Sem treino adicional. Critérios E023 mantidos: DE maior, direção >=95%, MMD/variograma <=102%, únicas>=95%, zeros a até 2pp do alvo. Só confirmar 1.024 se passar; repetir seeds 43/44 se confirmação passar. No máximo três hipóteses pequenas negativas neste novo ciclo.

**Código:** árvore herdada sem commit sobre 5be5ecb; fontes permitidas inalteradas; nenhuma leitura E10.5. Ollama solicitado somente para docstring de função NumPy.

**Triagem 512:** w=0,0625: DE 0,0857; direção 0,1784; MMD 0,07119; variograma 0,001340 (passa). w=0,125: 0,0952/0,1741/0,07068/0,001350 (falha direção). w=0,25: 0,1143/0,1653/0,06995/0,001379 (falha direção/variograma). Todos 512 únicos, zeros 90,739%. Somente w=0,0625 autorizado para confirmação 1.024. Ollama sugeriu docstring mecanicamente correta, revisada e resumida; teste confirma zeros, clipping, dtype e ausência de mutação. 19 testes focados passaram.

**Confirmação 1.024 E024:** DE=0,1468 (E021 0,0917), direção=0,1069 (0,1261), MMD=0,06112 (0,06169), variograma=0,001264 (0,001256). Apesar do ganho DE de 60%, direção regrediu 15,2%, falhando o gate. Rejeitado como candidata; sem geração E10.5, sem seeds adicionais. População 1.024 única e contrato válido. Próxima hipótese restringe correção aos genes mais fortes para reduzir perturbação do ranking global.

### E025 — Tendência somente em genes de maior mudança

**Hipótese pré-registrada:** E024 melhora DE, mas perturba o ranking de muitos genes de pequena mudança. Aplicar exatamente a mesma correção somente aos top-250 ou top-1.000 genes por mudança absoluta observada em D009→D010 preservará direção global. Nenhum gene selecionado pelo alvo.

**Configuração:** top={250,1000}, peso={0,125;0,25}, mesmos dados/seeds e gates E024. Quatro correções baratas de E021; sem treino adicional, encoder ou arquitetura novos. Escolher melhor DE entre aprovadas em 512 e confirmar somente essa em 1.024. Escopo permitido pela continuação do responsável. Implementação é uma máscara binária adicional sobre o delta de E024.

**Triagem E025 512:** top250/w0,125: 0,1048/0,1816/0,07083/0,001341; top250/w0,25: 0,1048/0,1815/0,07000/0,001355; top1000/w0,125: 0,1048/0,1792/0,07074/0,001347; top1000/w0,25: 0,1238/0,1750/0,06998/0,001371 (ordem DE/direção/MMD/variograma). Última falha direção/variograma. As três primeiras passam; empate DE resolvido pela menor intervenção: top250/w0,125, também melhor direção/variograma. Desempate explicitado após triagem, antes da confirmação; não foi critério originalmente pré-registrado. Confirmar top250/w0,125 em 1.024.

**Confirmação E025 1.024 seed42:** DE=0,1743; direção=0,1199; MMD=0,06074; variograma=0,001265; pb_rel_err=0,1055; JSD=0,0059; PB Pearson=0,9940. Passa gates, porém direção está muito próxima do limite de 0,119795 (95% de 0,1261). Ganho DE de 90,1%; direção -4,92%, MMD -1,54%, variograma +0,72%. 1.024 células únicas, zeros 90,662%. Iniciada estabilidade nas seeds de Transformer 43/44; programas/avaliação permanecem 42 e alvo/referência exatamente os E021 de 1.024.

**Estabilidade E025 1.024:** seeds42/43/44: DE 0,1743/0,1743/0,1560; direção 0,1199/0,1025/0,0955; MMD 0,06074/0,06265/0,06193; variograma 0,001265/0,001266/0,001268. Ganho DE consistente, gates conjuntos apenas seed42. Estatísticas completas em E025/stability.json. Não alegar estabilidade global. Novos treinos E021 de comparação usam epochs escolhidas 2 (índice zero), sem alteração de programas ou avaliação.

### E026 — Reduzir a competição das saídas auxiliares

**Hipótese pré-registrada:** E021 gera expressão com programas e âncoras; centros/dispensões latentes previstos não entram nessa geração. Reduzir o peso da loss estrutural de 1 para 0,1 pode liberar o Transformer para os programas sem adicionar complexidade. Mantém-se a mesma rede e todos os dados/seeds/escala E021. Uma única configuração nova, structural_weight=0,1; testar saída pura e correção E025 top250/w0,125. Critérios iguais E025; confirmar apenas se passar triagem 512. Não aumentar dimensão nem usar loss multicut. Comparação registra a alteração no critério de early stopping, que também usa o peso 0,1.

**Resultado:** na triagem fixa de 512 células, `structural_weight=0,1` obteve DE `0,0762`, direção `0,1598`, MMD `0,07309` e variograma `0,001380`, contra `0,0476/0,1867/0,07179/0,001332` do E021. Apesar do ganho de DE, falhou simultaneamente os gates de direção, MMD e variograma. Rejeitado sem correção E025, confirmação em 1.024 ou nova candidata.

### E027 — Kernel PCA aproximada e entrada celular crua

**Data:** 2026-09-25. **Marco:** M5/M7. **Seed:** 42.

**Hipótese e critério:** trocar scGPT por uma PCA não linear e remover os resumos de grupo pode conservar informação celular útil. Comparar no pseudo-holdout fixo E8.25→E8.5 do E021, com os mesmos 512 alvos e referência. Uma alternativa só melhora o estado atual se elevar DE sem regressão material de direção, MMD, variograma, diversidade ou unicidade.

**Método:** PCA de kernel RBF aproximada por 256 landmarks de Nyström, seguida de PCA para 32 dimensões, sobre 512 genes variáveis escolhidos apenas em D003–D010. O pre-image cru é uma regressão ridge por estatísticas suficientes. Foram comparados (a) K=32 resumos de proporção/média/dispersão e (b) 128 células individuais por estágio. A loss da variante celular combina média, dispersão e sliced Wasserstein e não usa pares célula-a-célula. Cada saída foi reconstruída por regressão crua, regressão + resíduo empírico da célula E8.25 mais próxima no latente, ou expressão do vizinho mais próximo.

| Entrada / decoder | DE ↑ | direção ↑ | MMD ↓ | variograma ↓ | variância | únicas |
|---|---:|---:|---:|---:|---:|---:|
| E021 atual | 0,0476 | 0,1867 | 0,07179 | 0,001332 | — | 512 |
| agrupada / cru | 0,0095 | 0,1384 | 0,13986 | 0,014779 | 0,072 | 512 |
| agrupada / resíduo | -0,0190 | 0,0382 | 0,09213 | 0,003624 | 0,923 | 512 |
| agrupada / vizinho | -0,0095 | -0,0170 | 0,07855 | 0,001386 | 0,945 | 508 |
| células cruas / cru | 0,0667 | 0,1003 | 0,18999 | 0,015067 | 0,064 | 512 |
| células cruas / resíduo | **0,0952** | 0,0576 | 0,12592 | 0,007172 | 0,921 | 512 |
| células cruas / vizinho | 0,0476 | 0,0069 | 0,08488 | 0,002762 | 0,977 | 174 |

**Comandos:** `uv run python -m src.scripts.run_e027_nonlinear_pca --output-dir outputs/population_transformer/E027/fixed_eval_seed42_n512 --seed 42 --n-cells 512 --device auto`; depois `uv run veckit --task T1` para cada uma das seis saídas contra os mesmos `reference.h5ad` e `target.h5ad`. Testes focados: `10 passed`; `compileall` e `git diff --check` passaram.

**Resultado e decisão:** a PCA não linear não resolveu o gargalo do decoder cru, que preservou apenas 6,4–7,2% da variância. O resíduo recuperou diversidade, mas não direção nem métricas distribucionais. Células cruas aumentaram o melhor DE para 0,0952, porém reduziram direção em 69% e pioraram MMD em 75% e variograma em mais de 5× frente ao E021. O vizinho cru colapsou para 174 células únicas. Rejeitar todas as variantes, não confirmar em 1.024 e manter E021 como candidata.

**Limitações:** uma configuração de kernel/dimensão, uma seed, somente 512 células e corpus temporal pequeno. O resultado rejeita esta implementação controlada; não prova que toda representação não linear ou todo Transformer de células falhará.

### E028 — Apenas dados oficiais e escala do Transformer

**Data:** 2026-09-25. **Marco:** M5/M7. **Seed:** 42.

**Hipótese:** retirar todas as fontes com sufixo `_ex` reduz mudança de domínio; aumentar a capacidade do Transformer pode compensar a perda de transições temporais.

**Limite experimental prévio:** sem `_ex`, existem apenas E8.5 e E9.5. Não há um terceiro estágio permitido para avaliar extrapolação. E028 mede somente capacidade de aprender E8.5→E9.5 em células reservadas do mesmo par de estágios. Não mede qualidade em E10.5 e não pode selecionar uma submissão.

**Dados e configuração:** somente D001/D002; nenhum caminho `_ex`. Por estágio: 1.024 células de treino, 256 de validação e 512 de teste, sem sobreposição. Nyström-RBF+PCA com 512 genes, 256 landmarks e 32 dimensões; decoder com resíduo empírico; loss populacional sem pares. Modelos: pequeno 64/2 camadas/4 cabeças (71.136 parâmetros), médio 128/4/4 (538.272), grande 256/6/8 (3.179.296). Baseline: deslocamento da média latente E8.5→E9.5 aprendido no treino.

| Modelo | DE ↑ | direção ↑ | MMD ↓ | variograma ↓ | pb_rel_err ↓ |
|---|---:|---:|---:|---:|---:|
| deslocamento médio | **0,6200** | **0,4759** | **0,03006** | **0,002763** | **0,0633** |
| Transformer pequeno | 0,4400 | 0,3182 | 0,04106 | 0,004546 | 0,0892 |
| Transformer médio | 0,4200 | 0,2806 | 0,05036 | 0,006458 | 0,1230 |
| Transformer grande | 0,3200 | 0,2246 | 0,11061 | 0,016895 | 0,2791 |

As perdas de validação também pioraram com a escala: `0,000643`, `0,002040` e `0,010315`. Todos os modelos geraram 512 células únicas e saídas válidas de 512 × 32.285.

**Comandos:** `uv run python -m src.scripts.run_e028_official_only --output-dir outputs/population_transformer/E028/seed42_n512 --seed 42 --device auto`; `uv run veckit --task T1` para as quatro saídas; testes focados, `compileall` e `git diff --check`.

**Decisão:** aumentar a complexidade não ajuda neste regime; piora todas as métricas de forma monotônica. O baseline simples é preferível para a transição conhecida. E028 não fornece evidência para remover `_ex` do treino da candidata E10.5, porque isso elimina toda validação temporal possível. Manter E021 e não gerar nova candidata.

### E029 — Transição oficial pareada por grupos compartilhados

**Data:** 2026-09-25. **Marco:** M5/M7. **Seed:** 42.

**Correção de escopo:** E028 comparou populações de células aleatórias. E029 implementa o desenho solicitado: um `MiniBatchKMeans(K=32)` é ajustado uma única vez sobre os latentes das células de treino E8.5+E9.5. Para cada ID `k`, o input contém proporção, centro e dispersão do grupo em E8.5; o alvo contém as mesmas estatísticas do grupo `k` em E9.5. Não existem pares de células. O par de grupos significa ocupação da mesma região latente compartilhada, não linhagem comprovada.

**Split e geração:** somente D001/D002. Por estágio, 1.024 células treinam embedding/K-means/Transformer e 256 selecionam checkpoint. O teste usa pool independente de 2.048 células E8.5 para gerar 512 células e 512 células E9.5 como alvo. A saída amostra âncoras conforme proporções previstas e aplica centro e dispersão futuros do respectivo grupo, seguida do decoder com resíduo empírico.

| Modelo | DE ↑ | direção ↑ | MMD ↓ | variograma ↓ | pb_rel_err ↓ |
|---|---:|---:|---:|---:|---:|
| copiar grupos E8.5 | 0,0645 | 0,2874 | 0,03487 | **0,001206** | 0,1052 |
| delta observado por grupo | **0,6129** | **0,5140** | 0,02336 | 0,002486 | **0,0703** |
| Transformer pequeno, 77.505 parâmetros | 0,4516 | 0,4778 | **0,01676** | 0,004058 | 0,0796 |
| Transformer médio, 550.977 parâmetros | 0,3387 | 0,4197 | 0,02230 | 0,004916 | 0,1147 |
| Transformer grande, 3.204.673 parâmetros | 0,1613 | 0,2323 | 0,02996 | 0,005172 | 0,1361 |

**Resultado:** o pareamento por grupos capturou mudança útil: tanto o delta direto quanto o Transformer pequeno superaram copiar E8.5 em DE, direção, MMD e pseudobulk. O Transformer pequeno obteve o melhor MMD, mas o delta direto venceu DE, direção e pseudobulk. Aumentar a rede piorou monotonicamente as métricas e a validação (`0,00613→0,01516→0,01739`). Todos os métodos produziram 512 células únicas e passaram o contrato.

**Decisão:** o desenho correto é o de grupos compartilhados, e não células aleatórias. Para apenas uma transição, usar o Transformer pequeno ou, preferencialmente como baseline, o delta direto por grupo. Não aumentar complexidade. O resultado continua sendo diagnóstico da transição conhecida e não demonstra extrapolação para E10.5.

**Geração E10.5 solicitada:** foram produzidas duas candidatas de 2.500 células sem leitura do alvo E10.5. Ambas usam um pool de 5.000 âncoras E9.5. `group_delta` reaplica ao estado E9.5 o delta por grupo observado E8.5→E9.5; `small` aplica ao estado E9.5 o checkpoint pequeno escolhido na época 246. Após decoder + resíduo, cada célula foi restaurada para `log1p(CP10k)`.

- `outputs/population_transformer/E029/final_seed42/e10_5_group_delta_2500.h5ad`: 2.500 únicas, 24 grupos, zeros `0,60732`, faixa `0–6,41452`, SHA-256 `ec9bb5299712a2dab32cd1b69064cf1204b2afdf11a4c9ab383249718733d3b6`.
- `outputs/population_transformer/E029/final_seed42/e10_5_small_2500.h5ad`: 2.500 únicas, 32 grupos, zeros `0,59886`, faixa `0–9,08759`, SHA-256 `0420d2c0137153dbc2a75da942a85ee12def6cbadd0e9a58af2f3bd4e12b496a`.

Os dois arquivos têm 2.500 × 32.285, ordem oficial, `float32`, valores finitos/não negativos e soma `expm1` por célula igual a 10.000 dentro de erro máximo `0,0049`. O máximo da variante `small` é alto em relação aos dados observados e deve ser tratado como risco de extrapolação. Esses arquivos são candidatos experimentais; não há avaliação local contra E10.5 oculto.

**Leaderboard informado pelo responsável:** `group_delta` obteve DE/direção/MMD/variograma `44,8/56,5/54,0/34,4` (ponderado `48,405`); `small` obteve `46,3/58,0/54,6/30,8` (ponderado `48,615`). Ambos melhoraram DE/direção/MMD frente ao E021, mas o caminho PCA+resíduo regrediu fortemente variograma.

### E030 — Decodificação esparsa ancorada em expressão

**Hipótese:** a densidade artificial do decoder PCA (`~40%` de entradas não zero, contra `~12%` em E9.5) é a causa principal da regressão do variograma. Manter PCA/K-means para prever grupos e proporções, mas aplicar o delta gênico E8.5→E9.5 somente em genes ativos de células E9.5 reais, deve preservar esparsidade e covariação.

**Configuração:** sempre três modelos — `group_delta`, Transformer `small` e Transformer `medium` — com as mesmas 5.000 âncoras E9.5, 2.500 saídas, seed 42 e escalas de expressão `0,125` e `0,25`. A saída é restaurada para `log1p(CP10k)`. Nenhum dado E10.5 foi lido.

**Validação estrutural:** as seis candidatas têm 2.500 células únicas, 32.285 genes, ordem oficial, `float32`, finitude e não negatividade. A fração de zeros ficou entre `0,87708` e `0,87854`, alinhada a E9.5/E021; máximos `6,8998` na escala 0,125 e `6,9883` na 0,25. Isso corrige o defeito estrutural mais provável do variograma, mas somente o leaderboard pode confirmar o ganho em E10.5.

**Decisão:** comparar sempre o trio delta/small/medium. Priorizar inicialmente escala 0,125 por ser a intervenção mais conservadora; manter 0,25 como segunda grade controlada. Artefatos em `outputs/population_transformer/E030/final_seed42/`.

**Leaderboard informado pelo responsável:** `small_anchored_0.125` obteve DE/direção/MMD/variograma `38,9/52,2/49,5/48,1` (ponderado `47,245`); `small_anchored_0.25` obteve `39,7/52,7/49,9/48,0` (ponderado `47,67`). Preservar todos os zeros recuperou o variograma de `30,8` no E029 small para aproximadamente `48`, mas perdeu a maior parte dos ganhos de DE, direção e MMD.

### E031 — Ativação seletiva de zeros por prevalência

**Hipótese pré-registrada:** E029 ativa zeros de forma quase global, enquanto E030 proíbe toda ativação. Permitir somente genes cuja prevalência aumentou de forma consistente dentro do mesmo grupo E8.5→E9.5, usando o padrão conjunto de uma célula doadora E9.5 do grupo, deve recuperar parte de DE/direção/MMD sem perder a esparsidade responsável pelo variograma de E030.

**Diagnóstico anterior à hipótese:** E9.5 tem fração de zeros `0,87779` e mediana de 3.878 genes expressos/célula. E029 `group_delta/small` reduziu zeros a `0,60732/0,59886`, elevou a mediana a `12.711/12.957,5` e aumentou prevalência em mais de 5 pontos percentuais para `22.369/22.481` genes. E030 escala 0,125 restaurou zeros a `0,87854/0,87708`, medianas `3.820/3.887,5` e deixou no máximo um gene com aumento de prevalência maior que 5 pontos. Portanto, os dois caminhos ocupam extremos opostos de ativação.

**Política fixada antes da execução:** somente D001/D002 e grupos K=32 de E029; seed 42; comparar obrigatoriamente `group_delta`, `small` e `medium` nas mesmas 512 células-alvo. Um gene é ativável no grupo apenas se `prevalência_E9.5 >= 0,10`, ganho de prevalência E8.5→E9.5 `>= 0,05` e delta médio positivo. Genes já expressos recebem escala `0,25`. Zeros elegíveis copiam conjuntamente o suporte de uma célula E9.5 do mesmo grupo, sem pareamento com o alvo, recebem `0,125 ×` a expressão doadora e são limitados aos 256 genes de maior confiança por célula. Fallback de grupo deve ser contado. A saída é renormalizada para `log1p(CP10k)`.

**Comparadores locais:** (1) decoder PCA + resíduo E029; (2) ancoragem rígida que preserva todos os zeros, sob as mesmas âncoras, proporções e escala expressa 0,25; (3) ativação seletiva. E9.5 reservado é alvo conhecido apenas da transição E8.5→E9.5; isto não mede E10.5.

**Gates pré-registrados para gerar 2.500 células:** contratos válidos e pelo menos 95% de células únicas nos três modelos; fração de zeros a no máximo 2 pontos percentuais do alvo; frente à ancoragem rígida, DE e direção não podem piorar e ao menos uma deve melhorar em pelo menos dois dos três modelos; MMD não pode piorar mais de 2% e variograma mais de 5% em nenhum modelo. Se qualquer gate comum falhar, registrar o resultado e não gerar E10.5/2.500.

**Configuração:** seed 42; `min_target_prevalence=0,10`; `min_prevalence_gain=0,05`; `expressed_scale=0,25`; `activation_scale=0,125`; `max_activations_per_cell=256`; sem E10.5 real, fonte externa ou pareamento celular.

**Resultado local em 512 células:**

| Modelo | Variante | DE ↑ | direção ↑ | MMD ↓ | variograma ↓ | zeros |
|---|---|---:|---:|---:|---:|---:|
| group_delta | rígida | 0,3548 | 0,4079 | 0,02467 | 0,000975 | 0,88030 |
| group_delta | seletiva | **0,4194** | **0,4307** | **0,02452** | **0,000837** | 0,87243 |
| small | rígida | 0,3226 | 0,4022 | 0,02363 | 0,000978 | 0,88101 |
| small | seletiva | **0,3710** | **0,4278** | **0,02349** | **0,000829** | 0,87321 |
| medium | rígida | **0,4032** | 0,4308 | 0,02271 | 0,000935 | 0,87844 |
| medium | seletiva | **0,4032** | **0,4526** | **0,02271** | **0,000831** | 0,87067 |

A variante seletiva melhorou direção e variograma nos três modelos, melhorou DE em dois e empatou no terceiro; MMD melhorou em dois e empatou, no arredondamento do scorer, no medium. Todas as nove saídas locais têm 512/512 células únicas e contrato válido. Frente à ancoragem rígida, o variograma caiu 14,2%, 15,2% e 11,1%, portanto nenhum gate comum falhou. A política ativou em média 250,7–253,9 zeros por célula e não usou fallback no delta; small/medium tiveram 4/6 fallbacks de doador no holdout.

**Geração E10.5 autorizada pelos gates:** o mesmo split de treino e a mesma política foram congelados; uma falha intermediária em que a contagem final alterava o split foi detectada antes do fechamento, corrigida separando `test_cells=512` de `n_cells=2500`, e os três artefatos foram regenerados. Nenhum dado E10.5 foi lido.

| Modelo | zeros | únicas | máximo | SHA-256 |
|---|---:|---:|---:|---|
| group_delta | 0,87107 | 2.500 | 6,9841 | `d8d4bec9953d6b6cd448fcc649d070dee8cf6927ef2b916f77c7420015646815` |
| small | 0,86992 | 2.500 | 6,9844 | `319957a971b53267dfae948584973e4c4add2b0de7d78c17a4878f165ed90875` |
| medium | 0,86990 | 2.500 | 6,9839 | `dd547a6e57ef368a9464b9ee7932cee68113e7acdfce1df39695c6ba309d13da` |

Os três arquivos têm 2.500 × 32.285, ordem oficial, `float32`, valores finitos/não negativos e erro máximo de soma CP10k `0,001953`. Artefatos, configurações e relatórios: `outputs/population_transformer/E031/seed42_k32_n512/` e `outputs/population_transformer/E031/final_seed42/`.

**Decisão:** E031 passa integralmente os gates locais pré-registrados e é uma candidata controlada para comparação externa, sempre como trio. O resultado local mede a transição conhecida e não demonstra desempenho em E10.5; não inferir propriedades do alvo oculto a partir de eventual leaderboard.

**Leaderboard E031 informado pelo responsável:** DE/direção/MMD/variograma `40,5/53,2/49,6/46,8`. A intervenção recuperou parte do DE frente ao E030, mas MMD permaneceu abaixo do E029 e o compromisso não justificou substituir a geometria densa como ponto de partida do próximo ciclo.

### E032 — Shrinkage da mudança do decoder somente em zeros

**Hipótese pré-registrada:** o E029 usa `raw_pred + (anchor_real - raw_anchor)`, equivalente a `anchor_real + (raw_pred - raw_anchor)`. Reduzir apenas o resíduo empírico em zeros seria incorreto e tenderia a aumentar a densidade, porque nesse suporte o resíduo é `-raw_anchor`. Em vez disso, manter a mudança completa nos genes expressos e multiplicar somente a mudança positiva do decoder nos zeros por beta deve preservar o sinal DE/distribucional do E029 com menos densificação.

**Fórmula:** `change = raw_pred - raw_anchor`; para `anchor_real > 0`, `output = max(anchor_real + change, 0)`; para `anchor_real == 0` com suporte no grupo E9.5, `output = beta * max(change, 0)`; sem qualquer expressão no grupo E9.5 de treino, manter zero. Renormalizar para `log1p(CP10k)`.

**Configuração fixada antes da execução:** somente D001/D002; K=32 e checkpoints E029; `beta={0,10;0,20}`; seed 42; 512 células; comparar obrigatoriamente `group_delta`, `small` e `medium` sob os mesmos splits. Sem E10.5 real, fonte externa ou pares celulares.

**Gates para gerar 2.500:** selecionar um único beta pelos resultados locais agregados. Em cada um dos três modelos, reter pelo menos 85% do DE e 90% da direção do decoder E029; MMD não piorar mais de 10%; variograma melhorar pelo menos 20%; pelo menos 95% de células únicas; contrato válido. Entre betas aprovados, escolher o de maior fração de zeros; empate pelo menor MMD médio. Se nenhum passar em todos os modelos, registrar e não gerar 2.500.

**Resultados em 512 células:**

| Modelo | Variante | DE ↑ | direção ↑ | MMD ↓ | variograma ↓ | zeros |
|---|---|---:|---:|---:|---:|---:|
| group_delta | E029 decoder | 0,6129 | 0,5124 | 0,02407 | 0,002418 | 0,6113 |
| group_delta | beta 0,10 | 0,5645 | 0,4381 | **0,02356** | **0,000775** | 0,7225 |
| group_delta | beta 0,20 | **0,5806** | **0,4499** | 0,02357 | 0,000861 | 0,7225 |
| small | E029 decoder | 0,4194 | 0,4631 | 0,01949 | 0,003545 | 0,5981 |
| small | beta 0,10 | **0,4355** | 0,4361 | **0,01858** | **0,000825** | 0,7332 |
| small | beta 0,20 | **0,4355** | **0,4450** | **0,01858** | 0,000967 | 0,7332 |
| medium | E029 decoder | 0,3226 | 0,4101 | 0,02853 | 0,004028 | 0,6105 |
| medium | beta 0,10 | **0,2903** | 0,4066 | **0,02488** | **0,001083** | 0,7421 |
| medium | beta 0,20 | **0,2903** | **0,4144** | **0,02512** | 0,001255 | 0,7421 |

Todos os arquivos têm 512/512 células únicas e contrato válido. Os dois betas produziram o mesmo suporte de zeros; beta controla amplitude, não quantidade de ativações. O beta 0,20 foi superior ou igual em retenção de DE/direção, mas `group_delta` reteve 87,80% da direção do decoder, abaixo do gate pré-registrado de 90%. Os demais gates passaram: DE do delta 94,73%; small melhorou DE/MMD; medium reteve 90,0% do DE e melhorou direção/MMD; variograma caiu 64,4–76,7%.

**Decisão:** resultado parcial promissor, mas nenhum beta passou todos os gates comuns por causa da direção do `group_delta`. Não gerar 2.500 células. A próxima hipótese pequena, se autorizada, é testar um único beta intermediário mais alto (por exemplo 0,30), pois 0,20 melhorou a retenção de direção sobre 0,10 enquanto ainda reduziu o variograma do delta em 64,4%; isso deve ser pré-registrado como novo experimento, não acrescentado retrospectivamente ao E032.

**Exceção operacional posterior:** o responsável solicitou explicitamente gerar apenas `small`, com 1.500 células, apesar de o gate comum do trio não ter passado. A candidata usa beta `0,20`, escolhido entre os dois valores já avaliados por melhor retenção de direção, e não altera a conclusão científica do E032. Ela deve ser rotulada como candidata autorizada fora do gate; não como configuração aprovada pelo protocolo original.

**Candidata excepcional gerada:** `outputs/population_transformer/E032/final_small_beta0.2_seed42_n1500/e10_5_small_zero_0.2_1500.h5ad`. Shape `1.500 × 32.285`, 1.500 células únicas, zeros `0,731214`, faixa `0–9,091075`, erro máximo CP10k `0,002930`, `float32`, finita, não negativa e genes na ordem oficial. SHA-256 `af981f701e1be115bdc38aa425b8c88e9f02661f61dc919a1d73975a80b8e756`. O máximo permanece alto e praticamente igual ao E029 small (`9,0876`), portanto o risco de valores extremos não foi corrigido por esta intervenção.

**Leaderboard da candidata excepcional informado pelo responsável:** DE/direção/MMD/variograma `45,5/54,9/56,1/43,1`. O MMD superou E029 small (`54,6`) e o variograma recuperou 12,3 pontos, com perdas moderadas de DE/direção frente a `46,3/58,0`.

### E033 — Treino multi-transição `_ex` e grade PCA × K

**Hipótese pré-registrada:** acrescentar as transições adjacentes D003–D011 fornece supervisão temporal ao Transformer agrupado, enquanto variar a dimensão da PCA e o número de grupos separa capacidade latente de granularidade populacional.

**Desenho:** `_ex` é supervisão adicional, não substitui D001/D002. Um Nyström-RBF+PCA e K-means compartilhados usam amostras de todos os estágios permitidos. As janelas `_ex` adjacentes e o split de treino oficial formam exemplos; validação/teste oficiais usam células separadas. Não há pares celulares. Geração/evaluação mantêm o shrinkage E032 beta 0,20.

**Etapas fixadas antes da execução:** (1) controle PCA=32/K=32 com `_ex`, comparado ao E032 official-only; (2) se operacional, grade `PCA={16,32,64} × K={16,32,64}`. Em cada ponto comparar `group_delta`, Transformer `small` e `medium`; large permanece excluído. Seed 42, mesmos 512 alvos oficiais e scorer completo. Seleção por métricas locais agregadas, nunca por leaderboard E10.5.

**Execução:** os nove pontos completaram treino, geração e scorer para o trio, totalizando 27 populações de 512 células. Todas tiveram 512 perfis únicos e passaram o contrato. O encoder/decoder/K-means compartilhados usaram até 512 células por `_ex`, mais splits oficiais separados; beta E032 fixo em 0,20.

**Resumo por melhor valor local de cada modelo:**

| Modelo | Melhor DE | Melhor direção | Melhor MMD | Melhor variograma |
|---|---|---|---|---|
| group_delta | PCA64/K64: 0,3710 | PCA64/K64: 0,4109 | PCA64/K64: 0,02012 | PCA64/K16: 0,000862 |
| small | PCA64/K16: 0,3226 | PCA32/K32: 0,3511 | PCA16/K64: 0,02347 | PCA32/K64: 0,001127 |
| medium | PCA64/K64: 0,3226 | PCA64/K32: 0,3798 | PCA16/K32: 0,02169 | PCA32/K64: 0,001252 |

O ponto mais coerente foi PCA64/K64 `group_delta`, que simultaneamente liderou DE, direção e MMD entre os deltas (`0,3710/0,4109/0,02012/0,000915`). Ainda assim, ele perde amplamente em DE/direção para E032 official-only beta 0,20 (`0,5806/0,4499`), embora melhore MMD (`0,02357→0,02012`). Nenhum Transformer `_ex` superou o small official-only em DE/direção, e as melhores configurações variam por métrica, sem uma região robusta da grade.

**Decisão:** rejeitar `_ex` como melhoria direta desta estrutura. A grade indica que maior PCA/K ajuda o delta, mas não resolve a mudança de domínio dos Transformers. Não gerar E10.5 a partir de E033 e preservar E032 small beta 0,20 como melhor candidata observada. Artefatos completos em `outputs/population_transformer/E033/pca{16,32,64}_k{16,32,64}_seed42_n512/`.

### E034 — Grade PCA × K official-only

Foram executados os nove pontos `PCA={16,32,64} × K={16,32,64}`, sempre com `group_delta`, `small`, `medium`, beta 0,20, seed 42 e os mesmos splits oficiais de 512 células. As 27 populações passaram contrato e unicidade.

O melhor compromisso Transformer foi novamente PCA32/K32 `small`: DE `0,5323`, direção `0,4296`, MMD `0,01852`, variograma `0,000934`. Frente ao E032 small anterior (`0,4355/0,4450/0,01858/0,000967`), o novo treino melhora fortemente DE, praticamente empata MMD/variograma e perde pouco em direção. PCA32/K16 teve melhor direção/MMD (`0,4629/0,01315`), mas DE `0,3548` e variograma `0,001687`; não é o melhor compromisso.

Nos baselines, PCA32/K32 `group_delta` manteve o maior DE (`0,5806`); PCA64/K16 teve maior direção (`0,4862`); PCA32/K64 teve melhor MMD/variograma (`0,02042/0,000751`). Não houve evidência de que PCA64/K64 seja superior para o Transformer official-only.

**Decisão:** preservar PCA32/K32 `small` como configuração Transformer selecionada localmente. A melhora vem do novo treino official-only, não de aumentar PCA/K. Não gerar automaticamente a candidata E10.5 nesta execução; checkpoint e artefatos estão em `outputs/population_transformer/E034/pca32_k32_seed42_n512/`.

### EXXX — Título curto

**Data:**

**Marco:**

**Responsável/agente:**

**Hipótese:**

**Critério de decisão definido antes da execução:**

**Código:** commit, branch ou descrição do estado da árvore.

**Dados:** IDs de `DATA_POLICY.md`, splits e filtros.

**Configuração:**

```yaml
seed:
model:
training:
generation:
```

**Comandos:**

```bash
# preencher
```

**Métricas:**

| Métrica | Baseline | Experimento | Observação |
|---|---:|---:|---|
| pseudobulk |  |  |  |
| direção/DE |  |  |  |
| MMD |  |  |  |
| covariação/variograma |  |  |  |

**Artefatos:**

- checkpoint:
- métricas:
- figuras:
- predição:

**Resultado:**

**Limitações:**

**Decisão:** manter, rejeitar, repetir ou modificar.

**Próximo experimento:**
