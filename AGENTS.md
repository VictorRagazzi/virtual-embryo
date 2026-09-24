# Instruções para agentes

## Missão

Construir e avaliar, de forma incremental e reproduzível, um modelo para a Task 1 do Virtual Embryo Challenge. O modelo deve prever uma **distribuição de células** em um estágio futuro, e não pares célula-a-célula nem uma célula média.

## Fontes de verdade

Antes de alterar código, leia:

1. `docs/SPEC.md`
2. `docs/ARCHITECTURE.md`
3. `docs/PLAN.md`
4. `docs/DATA_POLICY.md`
5. `docs/STATUS.md`

Consulte `docs/EXPERIMENTS.md` antes de repetir um experimento.

Em caso de conflito:

1. regras oficiais da competição;
2. `docs/DATA_POLICY.md`;
3. `docs/SPEC.md`;
4. `docs/ARCHITECTURE.md`;
5. `docs/PLAN.md`;
6. demais documentos.

## Regras de execução

- Execute apenas o marco marcado como atual em `docs/STATUS.md`.
- Não avance para o próximo marco enquanto os critérios de aceitação do atual não forem atendidos ou o bloqueio não for registrado.
- Faça mudanças pequenas, verificáveis e restritas ao marco.
- Preserve abordagens existentes; não apague nem reescreva experimentos anteriores sem solicitação explícita.
- Antes de implementar algo novo, procure utilidades equivalentes no repositório.
- Use `uv run` e as dependências declaradas em `pyproject.toml` sempre que possível.
- Toda execução deve ter seed explícita e configuração registrada.
- Não modifique arquivos `.h5ad` originais em `data/`.
- Não versione `.env`, chaves, dados brutos, checkpoints grandes ou predições pesadas.
- Evite densificar matrizes completas de aproximadamente 17 mil × 32.285 sem medir memória. Prefira leitura em blocos, matrizes esparsas ou subconjuntos quando possível.
- A saída oficial deve manter exatamente os 32.285 genes e a ordem liberada pela competição.
- A matriz final deve ser finita, não negativa, `float32` e estar na escala log-normalizada esperada.
- Nunca suponha correspondência célula-a-célula entre estágios.
- Rótulos celulares podem ser usados para diagnóstico, mas não devem ser necessários na saída nem tratados como verdade fornecida ao avaliador.

## Política para dados e modelos externos

- Não baixe, treine ou use uma fonte externa antes de registrá-la em `docs/DATA_POLICY.md`.
- Registre nome, versão, URL, licença, organismo, tecido, estágios e forma de uso.
- Pesos pré-treinados também são fontes de dados e exigem auditoria de proveniência.
- O único foundation model previsto para a nova abordagem é o scGPT. Não introduza Geneformer ou outro embedding sem decisão explícita do responsável pelo projeto.
- Se não for possível excluir que o checkpoint scGPT foi pré-treinado em estágios proibidos, marque-o como `BLOQUEADO_POR_PROVENIENCIA` e não o utilize em uma submissão oficial.
- Não use dados medidos na janela proibida definida pelas regras atuais da competição.

## Auxiliar local via Ollama

Há uma instância local do Ollama com Qwen 3.2 que pode ser usada para economizar chamadas ao agente principal em tarefas mecânicas e bem delimitadas, desde que a integração já esteja disponível no ambiente.

Uso permitido:

- gerar esqueleto de funções simples;
- sugerir testes unitários pequenos;
- escrever docstrings, tipos e boilerplate;
- explicar mensagens de erro não sensíveis;
- propor refatorações locais que serão revisadas.

Uso não permitido:

- decidir arquitetura, desenho experimental ou política de dados;
- escolher o melhor experimento ou interpretar resultados científicos sozinho;
- alterar arquivos sem revisão do agente principal;
- receber `.env`, chaves, segredos, dados brutos `.h5ad` ou grandes amostras de expressão;
- substituir testes, lint, revisão de diff ou critérios de aceitação;
- introduzir uma dependência obrigatória do Ollama no pipeline científico.

Toda contribuição do Qwen deve ser tratada como código não confiável: revisar, executar testes e registrar no experimento quando afetar código ou resultado. Use `OLLAMA_BASE_URL` e `OLLAMA_MODEL` se já estiverem configuradas; não invente porta nem nome do modelo e não bloqueie um marco se o serviço estiver indisponível.

## Validação obrigatória

Para cada marco:

1. execute testes ou comandos de validação;
2. salve métricas e configuração;
3. compare com pelo menos um baseline pertinente;
4. registre falhas e resultados negativos;
5. atualize `docs/EXPERIMENTS.md`;
6. atualize `docs/STATUS.md`.

Um script “rodar sem erro” não é evidência suficiente. Deve haver uma verificação numérica ou estrutural compatível com o objetivo do marco.

## Registro de experimentos

Cada experimento recebe um ID sequencial (`E001`, `E002`, ...). Registre:

- hipótese;
- commit ou estado do código;
- configuração completa;
- dados usados;
- seed;
- comandos executados;
- métricas;
- artefatos gerados;
- conclusão;
- decisão decorrente.

Não selecione uma configuração apenas por uma figura visual. Use métricas e critérios previamente declarados.

## Encerramento de uma tarefa

Ao concluir uma tarefa, reporte:

- arquivos modificados;
- comandos executados;
- resultados e métricas;
- critérios de aceitação atendidos ou não;
- limitações;
- próximo passo recomendado.
