# Especificação do projeto

## 1. Contexto

Este projeto é um trabalho final de disciplina e um ataque à **Task 1 — Temporal gene-expression distribution prediction** do Virtual Embryo Challenge.

| Split | Estágio | Papel |
|---|---:|---|
| Treino | E8.5 | população real observada |
| Treino | E9.5 | população real observada |
| Validação | E10.5 | alvo oculto avaliado no leaderboard |
| Teste | E12.5 | alvo final oculto |

Os estágios contêm células dissociadas de embriões diferentes. Não existe identidade celular compartilhada entre os dias.

## 2. Objetivo

Dados E8.5, E9.5 e um tempo alvo, produzir uma população sintética de células que represente a distribuição de expressão gênica no tempo alvo.

A saída deve ser um `.h5ad` com:

- linhas correspondendo a células sintéticas;
- 32.285 genes exatamente na ordem oficial;
- `.X` log-normalizado, finito, não negativo e preferencialmente `float32`;
- nenhuma dependência de rótulos celulares enviados pelo participante;
- número de células dentro dos limites oficiais vigentes.

Os limites de contagem devem ser lidos da versão atual do material oficial ou de `index.json`; não devem ser assumidos permanentemente no código.

## 3. Pergunta de pesquisa

Um Transformer que opera sobre resumos numéricos de populações celulares consegue aprender mudanças temporais de composição e estado, incluindo ramificações e surgimento de estados, e gerar uma população futura melhor que baselines simples?

## 4. Hipótese principal

Representar cada estágio como um conjunto de estados/grupos celulares, descritos por abundância, posição e variação em um espaço latente decodificável, torna a extrapolação mais adequada que o pareamento arbitrário célula-a-célula.

## 5. Escopo da primeira versão

A primeira versão deve:

1. auditar dados e reproduzir baselines;
2. extrair embeddings celulares com um checkpoint scGPT auditado;
3. treinar um decoder capaz de reconstruir expressão a partir desses embeddings;
4. criar agrupamentos conjuntos de E8.5 e E9.5;
5. resumir cada grupo numericamente;
6. treinar um Transformer temporal pequeno;
7. gerar células no espaço scGPT usando centros previstos e diversidade observada;
8. decodificar as células para o transcriptoma completo;
9. gerar e validar um `.h5ad` para E10.5.

## 6. Fora do escopo inicial

- Pareamento supervisionado célula-a-célula.
- Uso de uma LLM textual para decidir manualmente quais tipos celulares aparecem.
- Uso de Geneformer ou outro foundation model além de scGPT na nova abordagem, salvo mudança explícita de escopo.
- Predição direta dos 32.285 genes pelo Transformer temporal.
- Treinar simultaneamente todos os componentes antes de validar cada um isoladamente.
- Otimização exclusiva por aparência de UMAP.
- Uso de dados ou pesos de proveniência incompatível com as regras.

## 7. Métricas de interesse

As quatro perguntas oficiais da Task 1 são:

| Grupo de avaliação | Peso | O que exige do modelo |
|---|---:|---|
| recuperação de genes diferenciais | 25% | genes corretos mudando na direção correta |
| direção da mudança | 25% | ordenação da mudança no transcriptoma completo |
| distribuição de estados celulares | 30% | estados e proporções plausíveis, sem colapso |
| covariação gene-gene | 20% | relações entre genes preservadas entre células |

Durante desenvolvimento, usar `veckit` ou o scorer oficial quando aplicável, lembrando que alvos conhecidos são apenas pseudoavaliações e não estimam diretamente o score oculto.

## 8. Baselines obrigatórios

Toda proposta deve ser comparada, quando aplicável, com:

- `copy_last`: usar E9.5 sem alteração;
- `pseudobulk_shift`: deslocamento médio simples;
- aproximação PCA + delta de centroide já existente;
- abordagens prévias presentes em `src/approaches/`;
- reconstrução de referência do próprio encoder/decoder, que mede o teto técnico do gerador.

## 9. Critérios de sucesso do projeto

O projeto será considerado tecnicamente completo quando:

- o pipeline completo for reproduzível por configuração e seed;
- cada componente possuir teste isolado;
- houver comparação documentada contra baselines;
- o round-trip célula → latente → célula preservar diversidade e covariação de forma mensurável;
- a pseudoextrapolação em alvos conhecidos superar ou esclarecer por que não supera `copy_last`;
- o arquivo E10.5 passar no validador oficial;
- fontes externas e modelos pré-treinados estiverem auditados;
- falhas e limitações estiverem documentadas.
