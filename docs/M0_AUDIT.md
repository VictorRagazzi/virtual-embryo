# Relatório de auditoria M0

Gerado em 2026-09-24T00:32:48.650378+00:00 por `uv run python src/scripts/audit_m0.py --seed 42`. Os `.h5ad` foram lidos sem alteração e `X/data` foi varrido em blocos.

## Dados oficiais

| Arquivo | Shape | X | dtype | Valores armazenados | Densidade efetiva | Min–max | NaN/inf | float32 denso |
|---|---:|---|---|---:|---:|---|---|---:|
| E8.5 | 16787 × 32285 | csc_matrix | float32 | 70,979,922 | 12.067% | 0–6.89281 | False/False | 2.02 GiB |
| E9.5 | 17057 × 32285 | csc_matrix | float32 | 73,355,047 | 12.221% | 0–6.89828 | False/False | 2.05 GiB |

Concatenar ambos densamente requer 4.07 GiB (`float32`) ou 8.14 GiB (`float64`), antes de cópias temporárias. A PCA histórica não deve rodar integralmente.

Não existe `raw`, `layers` é vazio e `uns` contém apenas `celltype_palette`; portanto a escala log-normalizada é indício numérico, não confirmação de proveniência.

## Genes e metadados

Verificação automática: `same_length=True`, `same_order=True`, `first_mismatch_index=None`, 32.285 genes únicos. SHA-256 da sequência ordenada: `807549e15be019899fa83cbab864bec87ad278f60c272e39819c7302cc2285b8`. Primeiro/último: `Xkr4, Gm1992, Gm19938, Gm37381, Rp1` / `AC124606.1, AC133095.2, AC133095.1, AC234645.1, AC149090.1`.

E8.5: `obs=['_index', 'celltype']`, `var=['_index']`, `obsm=['X_umap.harmony.rna']`, `layers=[]`, `uns=['celltype_palette']`. E9.5 tem os mesmos conjuntos.

### Distribuição de `obs['celltype']`

| Rótulo | E8.5 | E9.5 |
|---|---:|---:|
| AVC-CM | 598 | 326 |
| BEC | 0 | 271 |
| Blood | 266 | 213 |
| EXEM | 863 | 0 |
| Endocardium | 0 | 1555 |
| Endothelium | 1139 | 0 |
| Epithelium | 0 | 233 |
| Foregut | 2405 | 871 |
| Hepatocyte | 0 | 1190 |
| IFT-CM | 949 | 690 |
| JCF | 1119 | 0 |
| LV-CM | 344 | 0 |
| NCC | 287 | 47 |
| NCC-derived | 0 | 621 |
| Neural Tube | 773 | 0 |
| OFT/RV-CM | 1176 | 1830 |
| Paraxial Mesoderm | 934 | 0 |
| Pericardium | 745 | 897 |
| Proepicardium | 0 | 883 |
| RV-CM | 442 | 0 |
| ST | 0 | 879 |
| SV-CM | 310 | 1273 |
| Surface Ectoderm | 1423 | 554 |
| V-CM | 0 | 1144 |
| aPHM | 0 | 226 |
| aSHF | 822 | 1050 |
| pPHM | 0 | 1194 |
| pSHF | 2192 | 1110 |

## Checkpoint scGPT e Ollama

`models/scGPT_heart/best_model.pt`: SHA-256 `23b4c43f403a4069e3acaa2a11c9a7be971b14e3c363739a6c4d24651c3feee7`, embedding `60697 × 512`, `embedding_dim=512`, 12 camadas/8 cabeças. `args.json` aponta apenas para `/scratch/ssd004/datasets/cellxgene/scb/heart/all_counts`. Situação: **BLOQUEADO_POR_PROVENIENCIA** — Não há URL, licença, hash de origem, corpus auditável ou intervalo de estágios que exclua a janela proibida.

Ollama: `OLLAMA_BASE_URL` configurada=False; `OLLAMA_MODEL` configurada=False. Nenhuma chamada ao serviço foi necessária.

## Repositório e reprodução

Divergências encontradas: `docs/ARCHITETURE.md` não tem o nome citado por AGENTS; o `get_dummy.py` histórico não gerava dummy; PCA densificava os dois estágios; o README referencia `src/scripts/umap_vec.py`, que não existe, e omite o `--input-scale` obrigatório de `pre_processement.py`; e testes referenciam os módulos ausentes `src.approaches.mouse_geneformer` e `src.scripts.subset_heart`. Abordagens históricas foram preservadas.

Comandos: `uv run python src/scripts/audit_m0.py --seed 42`; `uv run python src/scripts/get_dummy.py --reference data/E95.h5ad --output /tmp/ve-m0/dummy.h5ad --n-cells 2500 --seed 42`; `uv run python src/scripts/copy_last.py --input data/E95.h5ad --output /tmp/ve-m0/copy_last.h5ad --seed 42`; `uv run python src/scripts/predict_e10_5_PCA.py --e85 data/E85.h5ad --e95 data/E95.h5ad --out /tmp/ve-m0/pca.h5ad --max-cells-per-stage 128 --n-comps 30 --target-cells 1000 --skip-llm --seed 42`.

## Execução dos baselines

O dummy gerou 2.500 × 32.285 zeros `float32` e passou o contrato. `copy_last` reproduziu E9.5 integralmente (17.057 × 32.285; 73.355.047 valores armazenados) e passou o contrato. A PCA+delta rodou somente com 128 células por estágio, 30 componentes, `--skip-llm` e seed 42; explicou 0,308 da variância e gerou 128 células, portanto é diagnóstico e não submissão (abaixo do limite de contagem).

O `veckit` foi executado com E8.5/E9.5 pseudoamostrados estratificadamente a 128 células para não densificar os arquivos completos. No controle `copy_last` idêntico ao pseudoalvo: `pseudobulk_pearson=1,000`, `pb_rel_err=0`, `composition_JSD=0`, `variogram=0`; na PCA: `pseudobulk_pearson=0,992`, `pb_rel_err=0,1177`, `composition_JSD=0,0324`, `variogram=0,004970`. Esses valores não estimam E10.5 nem o leaderboard; a tabela completa está em E001.
