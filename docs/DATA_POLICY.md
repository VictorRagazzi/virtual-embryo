# Política de dados, pesos e proveniência

## 1. Finalidade

Este arquivo controla quais dados, modelos e recursos podem entrar no pipeline. Ele deve ser atualizado **antes** de qualquer download ou treinamento externo.

## 2. Regras operacionais

- Dados oficiais de treino E8.5 e E9.5 são permitidos.
- E10.5 e E12.5 são alvos retidos e não podem entrar no conteúdo da previsão enquanto forem splits de avaliação.
- Segundo as regras consultadas para a Task 1, dados externos depois de E9.5 até E13.5, inclusive, são excluídos; E10.5 e E12.5 são proibidos de forma absoluta enquanto forem alvos.
- Dados externos posteriores a E13.5 podem ser candidatos, desde que a fonte, estágio e uso sejam declarados e as regras atuais sejam reconfirmadas.
- Estágios informados por somitos ou Theiler stage devem ser convertidos e avaliados, não tratados como rótulos diferentes.
- Um modelo pré-treinado conta como fonte externa. Seu corpus deve ser auditado.
- Scores do leaderboard podem selecionar uma predição, mas não podem ser invertidos para recuperar propriedades do alvo.
- Reconsultar as regras oficiais antes de cada submissão, pois elas podem mudar.

## 3. Registro de fontes

Preencher uma linha antes do uso.

| ID | Fonte/modelo | URL/versão | Organismo/tecido | Estágios | Licença | Uso planejado | Proveniência verificada? | Situação |
|---|---|---|---|---|---|---|---|---|
| D001 | Challenge E8.5 | `data/E85.h5ad`, SHA-256 `8eab2d0ccaa89f92861b09816b76b6a731b8ed6b5995e4af196d9ed8ff76e504` | mouse, heart-centred | E8.5 | oficial | treino | sim | PERMITIDO |
| D002 | Challenge E9.5 | `data/E95.h5ad`, SHA-256 `0296e0043842f944a663f3c11ac1542d1bbee733c1cd657bd56f75af65958361` | mouse, heart-centred | E9.5 | oficial | treino/referência | sim | PERMITIDO |
| M001 | scGPT heart local | [perturblab/scgpt-heart](https://huggingface.co/perturblab/scgpt-heart), commit `f494d85028a4adb01487d9b53f2a6ee9da7765da`; SHA-256 local e LFS `23b4c43f403a4069e3acaa2a11c9a7be971b14e3c363739a6c4d24651c3feee7` | células cardíacas normais do CellxGene Census; card não traz manifesto por célula/estudo | aproximadamente 1,8 milhão de células; estágios individuais não manifestados | MIT declarada pelo card | único embedding celular da nova abordagem | evidência pública de fonte/escopo e identidade do binário; ausência de manifesto completo aceita como risco residual pelo responsável em 2026-09-24 | PERMITIDO_POR_DECISAO_DO_RESPONSAVEL |
| M002 | scGPT `whole-human` (candidato) | [Model Zoo oficial](https://github.com/bowang-lab/scGPT#pretrained-scgpt-model-zoo); objeto/version/hash ainda não selecionado | humano, células normais; múltiplos tecidos | não reportados no Model Zoo | MIT no repositório oficial; termos do objeto ainda confirmar | possível substituto de M001, somente após auditoria | parcialmente: descrição oficial informa 33 milhões de células humanas normais; falta manifesto/versionamento do objeto e análise de regra temporal | PENDENTE |
| D003 | Mouse Gastrulation Atlas, subconjunto local | `data/E65_ex.h5ad`, SHA-256 `35aa54490a6ef8fc37efc7a97ea33bf45f1ed1add7e684b21943e87423baf04c`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma nascente | E6.5 / TS9 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | origem/estágio/metadados compatíveis com E-MTAB-6967; transformação local não possui manifesto independente | PERMITIDO |
| D004 | Mouse Gastrulation Atlas, subconjunto local | `data/E675_ex.h5ad`, SHA-256 `9edd7afc8f82c07199f0dae78ad322e2773f525ede09cce318257f5bed37699f`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma | E6.75 / TS9 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | idem D003 | PERMITIDO |
| D005 | Mouse Gastrulation Atlas, subconjunto local | `data/E70_ex.h5ad`, SHA-256 `4be77ba03f32b493d578325cb4bf33ba7625ede232b71f040bcbdd002f01b9e8`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma | E7.0 / TS10 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | idem D003 | PERMITIDO |
| D006 | Mouse Gastrulation Atlas, subconjunto local | `data/E725_ex.h5ad`, SHA-256 `cc13e17f8f4a9f72423ef2f9d3f1fd4e6e54a4af316eedd51145b5f5795b64a9`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma | E7.25 / TS10 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | idem D003 | PERMITIDO |
| D007 | Mouse Gastrulation Atlas, subconjunto local | `data/E75_ex.h5ad`, SHA-256 `0e58ffac018e857a948d58dfbda5989a07ac2493fba2b4b5b6f352943899fd23`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma/cardiomiócitos | E7.5 / TS11 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | idem D003 | PERMITIDO |
| D008 | Mouse Gastrulation Atlas, subconjunto local | `data/E775_ex.h5ad`, SHA-256 `4dd53f735c90414a4d747ef93ccb548c667e1331dab5ac89ee7798d56501f232`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma/cardiomiócitos | E7.75 / TS11 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | idem D003 | PERMITIDO |
| D009 | Mouse Gastrulation Atlas, subconjunto local | `data/E80_ex.h5ad`, SHA-256 `e6edaf3319e033bdfa71232cdc6d0697f67cadd61003eee00c934882d02aa43e`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma faríngeo/cardiomiócitos | E8.0 / TS12 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | idem D003 | PERMITIDO |
| D010 | Mouse Gastrulation Atlas, subconjunto local | `data/E825_ex.h5ad`, SHA-256 `acfb95df7f2a34efad5deea6d8cc3acaea056d15407afb697e15d917701b11b9`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma faríngeo/cardiomiócitos | E8.25 / TS12 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | idem D003 | PERMITIDO |
| D011 | Mouse Gastrulation Atlas, subconjunto local | `data/E85_ex.h5ad`, SHA-256 `ff2e063bdf4e40e46130427141b1cb284bcd7b4613024811c029aa0d7721b12d`; Pijuan-Sala et al. 2019, E-MTAB-6967 | mouse, embrião; mesoderma faríngeo/cardiomiócitos | E8.5 / TS12 | CC0 no repositório público | treino exploratório M2; embeddings e agrupamento conjunto | idem D003; não há sobreposição de identificadores com D001 | PERMITIDO |

Situações permitidas:

- `PERMITIDO`
- `PENDENTE`
- `BLOQUEADO_POR_ESTAGIO`
- `BLOQUEADO_POR_LICENCA`
- `BLOQUEADO_POR_PROVENIENCIA`
- `USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO`
- `PERMITIDO_POR_DECISAO_DO_RESPONSAVEL`

### Decisão explícita sobre M001

Em 2026-09-24, o responsável aceitou formalmente o risco residual de proveniência e autorizou M001 para a candidata de submissão. A documentação pública descreve pré-treino em aproximadamente 1,8 milhão de células cardíacas normais do CellxGene Census; o hash, repositório e licença estão registrados acima. Não existe manifesto completo das células/estudos e, portanto, esta decisão não equivale a afirmar proveniência perfeita ou integralmente auditada. Artefatos novos não devem ser marcados automaticamente como não submetíveis apenas por derivarem de M001; continuam sujeitos às demais regras de dados, contrato e validação.

## 4. Checklist por dataset

- [ ] URL e versão registradas.
- [ ] Licença compatível.
- [ ] Organismo e tecido conhecidos.
- [ ] Estágio ou intervalo conhecido.
- [ ] Conversão de estágio documentada quando necessário.
- [ ] Ausência de E10.5/E12.5 confirmada.
- [ ] Janela proibida filtrada.
- [ ] Lista de genes registrada.
- [ ] Normalização registrada.
- [ ] Batch/replicatas registrados.
- [ ] Forma de entrada no modelo descrita.

## 5. Checklist por modelo pré-treinado

- [ ] Nome exato do checkpoint.
- [ ] Hash/versão.
- [ ] Repositório e licença.
- [ ] Descrição do corpus de pré-treinamento.
- [ ] Lista ou intervalo de estágios embrionários auditado.
- [ ] Possível presença da janela proibida investigada.
- [ ] Uso permitido confirmado ou modelo bloqueado.
- [ ] Função no pipeline registrada: encoder, embedding auxiliar ou baseline.

O projeto decidiu não utilizar Mouse Geneformer ou outro foundation model na nova abordagem. Referências a modelos anteriores podem permanecer no histórico do repositório, mas não devem entrar neste pipeline.

### Auditoria da série D003–D011

Os nove arquivos foram fornecidos e selecionados localmente pelo responsável. Todos têm exatamente 32.285 genes na ordem oficial (SHA-256 da sequência `807549e15be019899fa83cbab864bec87ad278f60c272e39819c7302cc2285b8`), valores `float32` finitos e não negativos e metadado `uns['log1p']`. Em amostra determinística de até 128 células por arquivo, `sum(expm1(X), genes)` é 10.000 com erro de arredondamento, confirmando `log1p(CP10k)`. Não há IDs `obs_names` ou `cell_velocyto_loom` repetidos entre os arquivos; tampouco há `obs_names` em comum com D001/D002. A filtragem local reteve apenas quatro rótulos mesodérmicos/cardiogênicos e não veio acompanhada de manifesto de transformação; por isso o uso é restrito aos experimentos exploratórios autorizados e essa seleção não deve ser confundida com o atlas integral.

## 6. Dados derivados

Artefatos derivados devem guardar:

- IDs das fontes;
- parâmetros de filtragem;
- genes removidos ou agrupados;
- seed;
- versão do código;
- data da geração;
- shape e dtype;
- hash ou manifesto dos arquivos de origem.

## 7. Observação sobre pré-processamento

O arquivo final precisa conter os 32.285 genes. Filtros de baixa variância ou agrupamentos de genes podem ser usados internamente, mas o decoder e a exportação devem restaurar a matriz completa na ordem oficial. Genes descartados do encoder não podem simplesmente desaparecer da submissão.
