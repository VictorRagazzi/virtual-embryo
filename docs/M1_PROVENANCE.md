# Gate de proveniência M1 — scGPT heart

**Data:** 2026-09-23

## Identidade confirmada

O SHA-256 local de `models/scGPT_heart/best_model.pt` é:

```text
23b4c43f403a4069e3acaa2a11c9a7be971b14e3c363739a6c4d24651c3feee7
```

Ele coincide com o objeto Git LFS publicado no commit [`f494d85028a4adb01487d9b53f2a6ee9da7765da`](https://huggingface.co/perturblab/scgpt-heart/commit/f494d85028a4adb01487d9b53f2a6ee9da7765da) de [`perturblab/scgpt-heart`](https://huggingface.co/perturblab/scgpt-heart). O card declara licença MIT, `bowang-lab/scGPT` como base/origem e que o reupload serve ao PerturbLab.

O `args.json` local coincide com o publicado e registra `data_source=/scratch/ssd004/datasets/cellxgene/scb/heart/all_counts`, `save_dir=.../cellxgene_census_heart-May15-15-54-2023`, 12 camadas, 8 cabeças, `embsize=512`, 51 bins e sequência máxima de 1.200 genes. A dimensão de embedding é, portanto, 512.

## Decisão

**Situação: `BLOQUEADO_POR_PROVENIENCIA`.** A identificação do binário e uma licença declarada não demonstram quais coleções/células entraram no pré-treino ou eventual treinamento heart-specific. Não há manifesto de organismos, estágios embrionários, filtros ou exclusão explícita de E10.5–E13.5. Logo não é possível excluir a janela proibida definida em `DATA_POLICY.md`.

Nenhum peso foi baixado durante a auditoria. A decisão exploratória inicial foi substituída em 2026-09-24 por decisão explícita do responsável: M001 pode ser usado na candidata de submissão. A evidência pública descreve aproximadamente 1,8 milhão de células cardíacas normais do CellxGene Census; a ausência de manifesto completo de células/estudos é aceita como risco residual. Isso não constitui alegação de proveniência perfeita ou integralmente auditada.

## Candidato registrado, ainda não usado

Foi registrado M002: `whole-human` do [Model Zoo oficial do scGPT](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo). A fonte o descreve como pré-treinado em 33 milhões de células humanas normais; a documentação oficial do corpus descreve uma construção baseada no CellXGene Census, com versão `2023-05-08` e filtro de células normais por tecido. [Model Zoo](https://github.com/bowang-lab/scGPT), [documentação do corpus](https://github.com/bowang-lab/scGPT/blob/main/data/cellxgene/README.md), [configuração do filtro](https://github.com/bowang-lab/scGPT/blob/main/data/cellxgene/data_config.py).

Isso ainda não libera M002: não há objeto/version/hash selecionado, manifesto do corpus ligado a esse objeto, nem decisão documentada sobre sua compatibilidade temporal e humano→camundongo. A situação é `PENDENTE`; nenhum download foi feito.

## Evidência consultada

- [Arquivo `args.json` publicado](https://huggingface.co/perturblab/scgpt-heart/blob/main/args.json).
- [Commit publicado com o hash LFS, licença declarada e descrição de origem](https://huggingface.co/perturblab/scgpt-heart/commit/f494d85028a4adb01487d9b53f2a6ee9da7765da).
