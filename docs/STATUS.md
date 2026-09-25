# Status do projeto

## Estado atual

**Marco ativo:** M5/M7 — E034 grade PCA × K official-only

**Situação:** E034_CONCLUÍDO_POSITIVO — grade 3×3 completa; small PCA32/K32 selecionado

**Última atualização:** 2026-09-25

## Objetivo do marco atual

Avaliar a mudança E8.5→E9.5 por pares de grupos K-means compartilhados, sem pareamento celular, usando somente dados oficiais.

## Próxima ação autorizada

Preservar o checkpoint E034 small PCA32/K32. Se autorizado, gerar uma candidata E10.5 com a mesma configuração e beta 0,20; não promover PCA64/K64, que não venceu no Transformer official-only.

## Não autorizado

- baixar datasets externos;
- integrar ou trocar checkpoints do scGPT sem atualização prévia de `DATA_POLICY.md`;
- usar E10.5 real para treino, seleção ou correção sem autorização e registro explícitos;
- reorganizar ou excluir abordagens existentes.

## Autorização operacional de 2026-09-24

O responsável autorizou explicitamente implementar e treinar o Transformer temporal apesar dos gates científicos negativos de M1/M3. Esses gates permanecem limitações documentadas, não bloqueios operacionais acadêmicos. E012/E013 permanecem diagnósticos vazados e não sustentam alegação de generalização. Não serão baixadas novas fontes nem introduzido outro foundation model.

## Decisão de proveniência de M001 em 2026-09-24

O responsável autorizou M001 para a candidata de submissão, aceitando explicitamente o risco residual decorrente da ausência de manifesto completo das aproximadamente 1,8 milhão de células cardíacas normais do CellxGene Census descritas publicamente. A proveniência não é considerada perfeita nem integralmente auditada; contudo, artefatos derivados deixam de ser automaticamente não submetíveis por esse único motivo.

## Fatos conhecidos

- treino oficial: E8.5 e E9.5;
- alvo de validação: E10.5;
- alvo final: E12.5;
- modalidade: scRNA, sem coordenadas espaciais;
- transcriptoma esperado: 32.285 genes;
- arquivos informados: `data/E85.h5ad` e `data/E95.h5ad`;
- ambiente Python gerenciado com `uv`;
- há abordagens históricas de PCA, clustering, scGPT/Mouse Geneformer e MOSCOT/OT, mas a nova abordagem usará apenas scGPT como fonte de embeddings;
- existe uma instância local do Ollama com Qwen 3.2 disponível como auxiliar opcional para tarefas simples de código; porta e tag exatas ainda devem ser confirmadas sem expor segredos;
- o scorer local usa alvo e referência conhecidos apenas como pseudoavaliação.

## Pontos que M0 deve confirmar

- [x] Os caminhos e nomes reais dos arquivos.
- [x] Shapes de E8.5 e E9.5.
- [ ] Escala e normalização efetivas — indício de log1p, sem metadado confirmatório.
- [x] Esparsidade e dtype.
- [x] Ordem e identidade dos genes.
- [x] Campos de `obs`, `var`, `obsm`, `layers` e `uns`.
- [x] Scripts que realmente existem versus descrição histórica.
- [x] Comandos que executam no ambiente atual.
- [x] Resultados dos baselines existentes.
- [x] Testes presentes e ausentes.
- [x] Estado do Git e mudanças não relacionadas que devem ser preservadas.

## Decisões abertas

| ID | Questão | Opções | Momento da decisão |
|---|---|---|---|
| Q001 | Arquitetura do decoder scGPT → expressão | começar pelo mais simples; adicionar perdas populacionais depois | após M0 |
| Q002 | Checkpoint/dimensão do scGPT | identificar e auditar a versão existente | M0/M1 |
| Q003 | Número de grupos | 50, 100 ou 200 | M2 |
| Q004 | Ajustar o scGPT ou mantê-lo congelado? | congelado inicialmente | M6 |
| Q005 | Dados temporais externos | definir após política e inventário | M4 |
| Q006 | Como acessar o Ollama local? | confirmar `OLLAMA_BASE_URL` e `OLLAMA_MODEL`; uso sempre opcional | M0 |

## Trabalho concluído

- M0 — Auditoria, reprodutibilidade e baselines: concluído em 2026-09-23. Ver `docs/M0_AUDIT.md`, `docs/M0_AUDIT.json` e E001.
- M1 — Gate de proveniência: identidade do checkpoint local confirmada em 2026-09-23; em 2026-09-24 o responsável aceitou formalmente o risco residual e autorizou M001 para candidata de submissão. Ver `docs/M1_PROVENANCE.md` e `docs/DATA_POLICY.md`.
- M1 — Round-trip: E007–E013 extraíram embeddings 512 e decodificaram 32.285 genes com encoder congelado. O decoder cru de E011 mantém variância de apenas 0,0760. E012/E013 são diagnósticos explicitamente vazados: E013 recupera o alvo por resíduo-oráculo, com métricas ~perfeitas, mas não mede generalização. Por autorização explícita do responsável, isto libera apenas M2 exploratório. Ver `docs/M1_ROUNDTRIP.md`.
- M2 — E014 auditou D003–D011, cacheou 44.084 CLS 512, comparou MiniBatchKMeans K=50/100/200 em seeds 42/43/44 e serializou tokens/resíduos. K=50 teve o maior ARI médio (0,4248), mas a estabilidade é apenas moderada. Ver `docs/M2_POPULATION_TOKENS.md`.
- M3 — E015 reextraiu embeddings compatíveis com E011 e reconstruiu E9.5 conhecido com K=50. Centros + resíduos produziram 506/512 expressões únicas, mas preservaram só 7,7% da variância e não superaram centroides em pseudobulk/MMD. Gate não atendido; ver `docs/M3_KNOWN_POPULATION.md`.
- M4/M5 — E016 transferiu resíduos de expressão E8.5→E9.5 e recuperou razão de variância 0,9517 com RFF-MMD 0,0106. E017 estabeleceu baselines temporais; E018 treinou Transformer/MLP no holdout D011, mas copy-last venceu no erro agregado (0,00372 vs. 0,00941 MLP e 0,01976 Transformer). Ver `docs/M5_TEMPORAL_TRANSFORMER.md`.
- M5/M7 exploratório — E019 reajustou o Transformer nas janelas D003–D011 e gerou três arquivos E10.5 de 512 × 32.285 sem consultar E10.5 real. Todos passam o contrato estrutural; a variante principal é decoder + resíduo E9.5. Artefatos: `/tmp/ve-m7/E019/`.
- M5/M7 — E020 executou a busca pesada e gerou três populações de 2.500 células. Todas as configurações perderam para `copy_last` no holdout; a melhor teve ganho relativo `-0,2861`. O retreino final antigo registrava o índice da época como validação, falha corrigida no início de E021.
- M5/M7 — E021 implementou a Estratégia A completa. No pseudo-holdout E8.25→E8.5 com 1.024 células, seed 42 e 32.285 genes, `α_expr=0,25`/`α_prop=0,25` melhorou simultaneamente DE `0→0,0917`, direção `0→0,1261`, MMD `0,06525→0,06169`, variograma `0,001260→0,001256` e erro de pseudobulk `0,1063→0,1049`. As seeds 43/44 recuperaram direção, mas não repetiram melhora conjunta, logo a estabilidade é parcial. A candidata E10.5 tem 2.500 células únicas e 87,72% de zeros.
- Leaderboard E021 informado pelo responsável: `de_score=38,8`, `de_direction=51,4`, `mmd_u=48,4`, `variogram=48,5`. Score ponderado pelas proporções da Task 1: `46,77`, contra `41,21` da submissão E020 anterior e `50,00` do copy-last. A melhora foi +0,9/+5,2/+9,6/+5,8 pontos por componente; distribuição e variograma ficaram próximos do baseline e direção o superou.
- M5/M7 — E022 adicionou loss assinada up/down nos cortes top-50/100/250/500/1.000 e testou pesos 0,05/0,1/0,2 no protocolo fixo de 512 células. O melhor peso 0,2 obteve DE `0,0190`, direção `0,1642`, MMD `0,07804` e variograma `0,001407`, todos piores que E021 exceto frente ao copy-last em direção. E022 foi rejeitado e não gerou nova candidata E10.5.
- M5/M7 — E026 reduziu o peso estrutural para 0,1. DE subiu para `0,0762`, mas direção/MMD/variograma regrediram; configuração rejeitada na triagem de 512 células.
- M5/M7 — E027 substituiu scGPT por Nyström-RBF + PCA 32D e comparou resumos K-means com células individuais. Células cruas + resíduo elevaram DE a `0,0952`, mas direção caiu a `0,0576`, MMD subiu a `0,12592` e variograma a `0,007172`; nenhuma variante superou E021. Ver `docs/E027_NONLINEAR_PCA_RAW_CELLS.md`.
- M5/M7 — E028 removeu todos os `_ex` e mediu E8.5→E9.5 em células reservadas. O deslocamento médio venceu Transformers pequeno/médio/grande; DE caiu `0,62→0,44→0,42→0,32` e MMD subiu `0,0301→0,0411→0,0504→0,1106`. O teste mede capacidade na transição conhecida, não E10.5. Ver `docs/E028_OFFICIAL_ONLY_CAPACITY.md`.
- M5/M7 — E029 corrigiu a unidade de treino para pares de grupos K=32 compartilhados. O delta direto por grupo obteve DE/direção `0,6129/0,5140`; o Transformer pequeno `0,4516/0,4778` e melhor MMD `0,01676`. Médio/grande pioraram. Ver `docs/E029_GROUPED_OFFICIAL_TRANSITION.md`.
- M5/M7 — E029 gerou duas candidatas E10.5 de 2.500 células, `group_delta` e `small`, a partir de âncoras E9.5 e sem acesso ao alvo. Ambas passam contrato e `log1p(CP10k)`; a variante `small` tem máximo alto (`9,0876`) e é a extrapolação mais agressiva.
- Leaderboard E029 informado pelo responsável: delta `44,8/56,5/54,0/34,4`; small `46,3/58,0/54,6/30,8` para DE/direção/MMD/variograma. Ponderados `48,405/48,615`; variograma é o único componente muito abaixo do baseline.
- M5/M7 — E030 substituiu o decoder final por células E9.5 ancoradas e delta gênico esparso, sempre para delta/small/medium. Seis candidatas de 2.500 células passam contrato; zeros retornaram a 87,7–87,9%, contra 59,9–61,5% no decoder PCA.
- Leaderboard E030 small informado pelo responsável: escala 0,125 obteve `38,9/52,2/49,5/48,1` e escala 0,25 `39,7/52,7/49,9/48,0` em DE/direção/MMD/variograma. O variograma foi recuperado, mas DE/direção/MMD regrediram frente ao E029.
- M5/M7 — E031 permitiu ativação de no máximo 256 zeros/célula, somente por ganho de prevalência dentro do mesmo grupo e suporte conjunto de doador E9.5. Em 512 células, direção e variograma melhoraram nos três modelos frente à ancoragem rígida; DE melhorou em delta/small e empatou no medium. Os gates passaram e o trio final de 2.500 células tem 86,99–87,11% de zeros, 2.500 perfis únicos e contrato válido.
- Leaderboard E031 informado pelo responsável: `40,5/53,2/49,6/46,8` em DE/direção/MMD/variograma. DE foi parcialmente recuperado, mas MMD continuou abaixo do E029.
- M5/M7 — E032 voltou à geometria E029 e reduziu apenas a mudança positiva do decoder em zeros suportados. Beta 0,20 manteve 72,2–74,2% de zeros, melhorou fortemente variograma e preservou/melhorou MMD, mas `group_delta` reteve somente 87,8% da direção contra gate de 90%. Resultado parcial; nenhuma candidata de 2.500 foi gerada.
- Leaderboard da candidata excepcional E032 small beta 0,20: `45,5/54,9/56,1/43,1`; melhor MMD observado e forte recuperação de variograma frente ao E029.
- M5/M7 — E033 treinou o trio em nove combinações PCA×K usando transições `_ex`. PCA64/K64 `group_delta` foi o melhor compromisso `_ex` (`0,3710/0,4109/0,02012/0,000915` local), mas perdeu DE/direção para official-only. Nenhum Transformer `_ex` superou E032; resultado negativo por mudança de domínio.

## Bloqueios

- E023: 32+16 e 32+32 DE pioraram DE/direção/MMD/variograma frente ao E021; 64 gerais é inviável na matriz atual de apenas 37 deltas. Sem nova candidata, sem confirmação 1.024 ou estabilidade 43/44, pois o gate inicial falhou. Ver `docs/E023_HYBRID_PROGRAMS.md`.

- M001 / scGPT heart local: permitido por decisão explícita do responsável; a ausência de manifesto completo permanece risco residual documentado, não bloqueio automático.
- Normalização oficial: os dois `.h5ad` não trazem metadado que confirme o método/fator; a faixa é somente compatível com log1p.
- Suíte histórica: `tests/test_mouse_geneformer.py` e `tests/test_subset_heart.py` não coletam porque os módulos que importam não existem na árvore atual. O contrato M0 novo passa, mas o problema deve ser decidido antes de tomar a suíte completa como gate.
- Gate científico M1: a melhor configuração sem vazamento preserva apenas 12,8% da variância média por gene da validação (E009); E011 com perda de covariância atingiu somente 7,60%. E012/E013 são vazados e não resolvem esse gate. Por autorização explícita de 2026-09-24, isso permanece limitação científica, mas não bloqueia o Transformer exploratório.
- Gate científico M3: centros + resíduos preservam a variância latente, porém E011 reduz a razão de variância final a 0,0772 e não melhora pseudobulk/MMD contra centroides. Por autorização explícita de 2026-09-24, isso permanece limitação científica, mas não bloqueia o pipeline acadêmico ponta a ponta.
