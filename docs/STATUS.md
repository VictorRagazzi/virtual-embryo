# Status do projeto

## Estado atual

**Marco ativo:** M0 — Auditoria, reprodutibilidade e baselines

**Situação:** NÃO INICIADO

**Última atualização:** preencher ao instalar estes documentos no repositório.

## Objetivo do marco atual

Inspecionar o repositório e os dados reais, validar contratos e reproduzir baselines antes de implementar o novo encoder/decoder.

## Próxima ação autorizada

Executar o prompt de `prompts/START_AGENT.md`.

## Não autorizado ainda

- implementar o Transformer temporal;
- treinar o decoder dos embeddings scGPT;
- baixar datasets externos;
- integrar ou trocar checkpoints do scGPT;
- gerar submissão E10.5 com a nova arquitetura;
- reorganizar ou excluir abordagens existentes.

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

- [ ] Os caminhos e nomes reais dos arquivos.
- [ ] Shapes de E8.5 e E9.5.
- [ ] Escala e normalização efetivas.
- [ ] Esparsidade e dtype.
- [ ] Ordem e identidade dos genes.
- [ ] Campos de `obs`, `var`, `obsm`, `layers` e `uns`.
- [ ] Scripts que realmente existem versus descrição histórica.
- [ ] Comandos que executam no ambiente atual.
- [ ] Resultados dos baselines existentes.
- [ ] Testes presentes e ausentes.
- [ ] Estado do Git e mudanças não relacionadas que devem ser preservadas.

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

Nenhum marco concluído ainda.

## Bloqueios

Nenhum bloqueio técnico registrado. Proveniência do checkpoint scGPT permanece pendente.
