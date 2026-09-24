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
| D001 | Challenge E8.5 | preencher caminho/versão | mouse, heart-centred | E8.5 | oficial | treino | sim | PERMITIDO |
| D002 | Challenge E9.5 | preencher caminho/versão | mouse, heart-centred | E9.5 | oficial | treino/referência | sim | PERMITIDO |
| M001 | scGPT | preencher checkpoint e corpus | preencher | preencher | preencher | único embedding celular da nova abordagem | não | PENDENTE |

Situações permitidas:

- `PERMITIDO`
- `PENDENTE`
- `BLOQUEADO_POR_ESTAGIO`
- `BLOQUEADO_POR_LICENCA`
- `BLOQUEADO_POR_PROVENIENCIA`
- `USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO`

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
