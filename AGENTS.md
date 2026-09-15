# Orientações para agentes

Este é um projeto de uma disciplina de graduação. O código será lido por pessoas com pouca experiência, então clareza vale mais que abstração ou sofisticação.

## Contexto sob demanda

- Não leia o repositório inteiro antes de começar.
- Se a tarefa não exigir contexto do projeto, trabalhe apenas nos arquivos diretamente envolvidos.
- Se precisar entender o projeto, leia primeiro `ARCHITETURE.md`.
- Só leia o código-fonte quando `ARCHITETURE.md` não responder ao que a tarefa exige.
- Ao ler código, abra apenas os arquivos relacionados à mudança. Expanda o contexto somente quando uma dependência real aparecer.

## Como implementar

- Escolha a solução correta mais simples de explicar.
- Prefira código linear, como uma história: entrada, transformação, resultado.
- Organize as etapas na ordem em que acontecem.
- Evite criar muitos arquivos, camadas, classes, helpers ou abstrações pequenas.
- Mantenha uma função no arquivo atual quando separá-la não melhorar claramente a leitura.
- Use nomes completos e diretos. Comentários devem explicar o motivo, não repetir o código.
- Reutilize bibliotecas e padrões que já existem no projeto antes de adicionar dependências ou estruturas novas.
- Não prepare arquitetura para necessidades hipotéticas.
- Preserve os padrões do trecho alterado, salvo quando eles forem a causa do problema.

## LLM e trabalho determinístico

- Use scripts para tarefas com uma resposta reproduzível, como cálculos, transformações de dados, validações e geração de arquivos.
- Use LLM somente quando houver interpretação, conhecimento semântico ou geração de conteúdo.
- Se uma etapa misturar os dois casos, deixe o fluxo determinístico no script e isole apenas a decisão semântica.
- Antes de alterar essas áreas, consulte as seções correspondentes de `ARCHITETURE.md`.

## Entrega

- Faça somente mudanças necessárias para a solicitação.
- Mudança de comportamento deve incluir um teste que falhe sem a correção.
- Para documentação ou formatação, valide links, caminhos, comandos e consistência com a árvore atual.
- Execute apenas os testes e verificações relacionados aos arquivos alterados.
- Não altere nem reverta trabalho local que não pertence à tarefa.
- Ao finalizar, informe de forma curta: arquivos alterados, validações executadas e qualquer limitação real.
