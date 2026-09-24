# Plano de execução por marcos

## Regra geral

Somente um marco fica ativo por vez. Um marco termina quando seus critérios de aceitação são atendidos ou quando um bloqueio está documentado com evidências.

## M0 — Auditoria, reprodutibilidade e baselines

### Objetivo

Entender o estado real do repositório e estabelecer referências reproduzíveis antes de criar novos modelos.

### Tarefas

- inspecionar `pyproject.toml`, scripts e abordagens existentes;
- localizar E8.5 e E9.5 e validar os `.h5ad`;
- registrar forma, tipo, esparsidade, normalização, genes, metadados e uso de memória;
- confirmar ordem idêntica dos 32.285 genes;
- executar ou reparar o baseline `copy_last`;
- executar, se disponível, pseudobulk shift e PCA + delta de centroide;
- verificar a integração com `veckit`/scorer;
- criar testes pequenos para contrato do arquivo final;
- atualizar a documentação com fatos encontrados.

### Critérios de aceitação

- relatório de auditoria salvo;
- dados carregados sem modificação;
- gene order verificada automaticamente;
- ao menos `copy_last` reproduzido;
- comandos reproduzíveis documentados;
- nenhuma chave ou dado bruto incluído no versionamento.

## M1 — Embedding scGPT e decoder

### Objetivo

Extrair embeddings com scGPT, treinar o caminho inverso e medir quanta informação populacional é perdida no round-trip.

### Tarefas

- auditar e fixar o checkpoint scGPT;
- implementar extração reproduzível de embeddings;
- implementar um decoder mínimo configurável;
- criar splits que evitem vazamento entre treino e avaliação;
- reconstruir células conhecidas;
- comparar expressão real e reconstruída;
- medir pseudobulk, distribuição e covariação;
- salvar exemplos e métricas;
- manter o scGPT congelado na primeira versão.

### Critérios de aceitação

- saída tem 32.285 genes na ordem correta;
- valores finitos e não negativos;
- reconstruções não colapsam para uma célula média;
- diversidade e covariação são reportadas;
- desempenho supera o baseline de repetir o perfil médio;
- custo de memória, cache de embeddings e tempo é registrado;
- existe comando único de treino e comando único de avaliação.

### Gate

Não iniciar Transformer se o round-trip scGPT → decoder destruir a distribuição ou a covariação. Não introduzir outro encoder sem decisão explícita.

## M2 — Agrupamento conjunto e tokens populacionais

### Objetivo

Construir um vocabulário de estados e resumos numéricos por estágio.

### Tarefas

- concatenar latentes de E8.5 e E9.5;
- testar K-means com K inicial em 50, 100 e 200;
- medir estabilidade entre seeds;
- identificar grupos minúsculos ou vazios;
- calcular proporção, centro, dispersão, programas e resíduos;
- gerar relatório com top genes para interpretação;
- serializar o vocabulário e os resumos.

### Critérios de aceitação

- cada célula possui atribuição reproduzível;
- proporções por estágio somam 1;
- nenhum grupo problemático é ignorado silenciosamente;
- estabilidade entre seeds é reportada;
- dimensões e contrato dos tokens possuem testes;
- top genes são diagnóstico, não a única representação.

## M3 — Gerador populacional em alvo conhecido

### Objetivo

Validar a geração de células antes de adicionar previsão temporal.

### Tarefas

- reconstruir uma população conhecida a partir de seus resumos reais;
- amostrar grupos segundo proporções reais;
- gerar latentes usando centros + resíduos empíricos;
- decodificar para expressão completa;
- comparar população gerada com população real;
- testar fallback para grupo sem resíduos;
- medir sensibilidade ao número de células geradas.

### Critérios de aceitação

- geração produz células diferentes entre si;
- proporções amostradas se aproximam das solicitadas;
- arquivo gerado passa no contrato estrutural;
- MMD/covariação e estatísticas por gene são reportadas;
- gerador supera a repetição de centroides;
- falhas por grupo são registradas.

### Gate

Não treinar dinâmica temporal sobre um gerador que não reproduz adequadamente um estágio conhecido.

## M4 — Corpus temporal permitido e tarefas de treino

### Objetivo

Criar supervisão temporal suficiente sem violar as regras da competição.

### Tarefas

- inventariar séries externas candidatas;
- preencher `DATA_POLICY.md` para cada fonte;
- bloquear automaticamente estágios proibidos;
- harmonizar genes e normalização sem misturar alvos ocultos;
- criar exemplos do tipo `(estágios anteriores, tempo alvo) → estágio conhecido`;
- separar treino, validação e teste temporal por fonte/estágio;
- auditar o checkpoint scGPT antes de uso oficial.

### Critérios de aceitação

- toda fonte possui proveniência e licença;
- nenhuma célula de janela proibida entra no treino;
- splits temporais são reproduzíveis;
- não há target leakage na construção de vocabulário para a avaliação declarada;
- existe relatório de cobertura de genes e estágios.

## M5 — Transformer temporal mínimo

### Objetivo

Prever resumos de uma população futura em tarefas com verdade conhecida.

### Tarefas

- implementar Transformer numérico pequeno;
- prever proporções, centros e dispersões;
- treinar primeiro em tarefas externas permitidas;
- realizar pseudo-holdout temporal;
- comparar contra `copy_last`, extrapolação linear e PCA shift;
- executar ablações simples;
- gerar células a partir dos resumos previstos e avaliar o pipeline completo.

### Critérios de aceitação

- treinamento e inferência são reproduzíveis;
- não há NaN, massas negativas ou proporções inválidas;
- métricas são reportadas no espaço de resumos e no espaço de células;
- comparação contra baselines usa o mesmo alvo e referência;
- ao menos uma hipótese de melhoria é sustentada, ou o resultado negativo é documentado;
- o modelo não é escolhido apenas pelo loss de treino.

## M6 — Refinamento controlado do scGPT e decoder

### Objetivo

Determinar se mudanças controladas no uso do scGPT ou no decoder melhoram o pipeline sem destruir a geração.

### Tarefas

- reconfirmar a proveniência dos pesos;
- comparar decoder simples e decoder com perdas populacionais;
- testar ajuste parcial do scGPT somente se permitido e necessário;
- manter dados, splits e orçamento equivalentes;
- medir ganho, custo e estabilidade.

### Critérios de aceitação

- proveniência aprovada para uso oficial;
- ablação controlada registrada;
- configuração e checkpoint explicitamente registrados;
- nenhuma regressão silenciosa em reconstrução ou geração.

## M7 — Predição E10.5

### Objetivo

Gerar uma submissão de validação completa.

### Tarefas

- congelar configuração escolhida sem usar o score para inferir propriedades do alvo;
- treinar com dados permitidos;
- inferir resumos para E10.5;
- gerar múltiplas sementes de população;
- aplicar verificações estruturais e biológicas;
- validar o `.h5ad` com ferramenta oficial;
- selecionar submissão apenas segundo protocolo documentado.

### Critérios de aceitação

- arquivo passa em todas as validações;
- genes e ordem são exatos;
- valores são finitos, não negativos e log-normalizados;
- contagem está na faixa atual da competição;
- não há células duplicadas em massa nem colapso evidente;
- artefatos, configuração e seed são preservados;
- método e fontes externas estão prontos para divulgação.

## M8 — Extensão para E12.5

### Objetivo

Após a liberação permitida de E10.5 na fase final, atualizar e avaliar a extrapolação para E12.5 conforme as regras então vigentes.

### Critérios

Definir somente quando a fase e os dados oficiais mudarem. Revalidar todas as regras antes de usar E10.5 como treino.
