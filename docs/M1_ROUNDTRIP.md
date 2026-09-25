# Round-trip M1 — scGPT congelado e decoder

**Data:** 2026-09-24

## Escopo e reprodutibilidade

M001 é usado exclusivamente como `USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO`. O encoder scGPT de 12 camadas e dimensão 512 foi carregado sem chaves faltantes e congelado. Cada ablação usou 64 células de E8.5 e 64 de E9.5, split estratificado por estágio (102 treino, 26 validação), seed 42, 256 genes de maior variância e quatro threads PyTorch.

Os símbolos murinos foram resolvidos somente por igualdade exata ou caixa alta no vocabulário; houve 16.152 correspondências por caixa alta. Não foi usado mapeamento externo de ortólogos. A sequência contém `<cls>` mais 256 genes; o decoder sempre reconstrói os 32.285 genes na ordem oficial.

```bash
uv run python src/scripts/run_scgpt_roundtrip.py \
  --e85 data/E85.h5ad --e95 data/E95.h5ad \
  --output-dir /tmp/ve-m1/E007_final \
  --cells-per-stage 64 --input-genes 256 --epochs 8 \
  --embedding-batch-size 8 --decoder-batch-size 16 \
  --learning-rate 0.001 --validation-fraction 0.2 \
  --torch-threads 4 --seed 42
```

Os artefatos temporários (`decoder.pt`, arrays de validação, `config.json`, `metrics.json`) permanecem em `/tmp/ve-m1/` e não são versionados. As matrizes completas dos estágios continuam esparsas; em cada execução, a única densificação persistente de treino é o lote de 102 células por 32.285 genes (`float32`, ~12,6 MiB).

## Resultados

| ID | Decoder | Perda de variância | MSE decoder | MSE perfil médio | Pearson pseudobulk | Razão de variância | Spearman variância gênica | Pearson covariância amostrada |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| E007 | linear + Softplus | 0 | 0,061462 | 0,064709 | 0,97737 | 0,06750 | 0,71896 | 0,88253 |
| E008 | MLP 512 + Softplus | 5 | 0,076016 | 0,064709 | 0,95281 | 0,26602 | 0,69509 | 0,63033 |
| E009 | linear + Softplus | 1 | 0,062365 | 0,064709 | 0,97681 | 0,12832 | 0,71985 | 0,82723 |
| E010 | linear + Softplus, 512 genes | 0 | 0,061478 | 0,064709 | 0,97739 | 0,06093 | 0,71945 | 0,87559 |
| E011 | linear + Softplus, covariância normalizada em 128 genes | 0 | 0,061889 | 0,064709 | 0,97580 | 0,07601 | 0,71896 | 0,89252 |
| E012 | E011 + projeção no vizinho E9.5 | 0 | 0,095270 | 0,064709 | 0,91757 | 0,39456 | 0,84223 | 0,70964 |
| E013 | E011 + resíduo-oráculo da validação | 0 | ~0 | 0,064709 | ~1 | ~1 | ~1 | 1 |

Todas as matrizes previstas têm shape `26 × 32.285`, são `float32`, finitas e não negativas. E007 reduziu MSE em 5,0% contra repetir o perfil médio; E009 reduziu 3,6%. Porém, a melhor razão de variância é apenas 0,266 (E008), e esta configuração piora MSE e covariância. A configuração com melhor compromisso de MSE (E009) preserva somente 12,8% da variância. E010 elevou a entrada a 512 genes, sem recuperar diversidade (0,0609), indicando que a limitação não é apenas a cobertura de genes. E011 manteve o decoder linear e adicionou uma perda de covariância normalizada (diagonal e termos fora da diagonal) nos 128 genes mais variáveis de cada batch, com peso 0,1. Em ~21 s, elevou a razão de variância de E007 de 0,0675 para apenas 0,0760 e a covariação amostrada a 0,8925; o ganho de diversidade é insuficiente e fica abaixo de E009.

E012 e E013 foram explicitamente autorizados pelo responsável como diagnósticos com vazamento para destravar a continuação acadêmica. E012 projeta a saída no vizinho de expressão E9.5, incluindo referência de validação: aumentou a razão de variância a 0,3946, mas piorou MSE para 0,095270 e usou só seis referências. E013 soma o resíduo conhecido da própria validação; portanto reconstitui o alvo por identidade (MSE ~0, variância/covariância ~1). Esses resultados confirmam o contrato do caminho completo, mas não corrigem o decoder cru de E011 nem são evidência de generalização ou candidatos a submissão.

## Decisão

O critério científico M1 de evitar colapso de diversidade **não foi atendido** pelo decoder sem vazamento. Por autorização explícita do responsável, o marco é considerado suficiente apenas para continuar M2 de modo exploratório após o diagnóstico-oráculo E013. M3 e qualquer Transformer temporal continuam não autorizados por este documento; antes de implementar ou treinar Transformer, o responsável deve ser avisado conforme combinado.

Tentativas anteriores com 1.199 genes de entrada não completaram no limite interativo de CPU e foram interrompidas; não geraram artefatos aceitos. A correção de execução foi fixar quatro threads e limitar o primeiro experimento a 256 genes, explicitamente registrado acima.

Próximo passo: iniciar M2 exploratório com agrupamento conjunto dos embeddings já extraídos, mantendo claro que o decoder cru ainda é limitado e que E012/E013 não são candidatos a uso externo ou submissão.
