# Proposta de pareamento por transporte ótimo

O primeiro experimento recomendado abaixo está implementado como `--pairing ot`;
o padrão continua sendo o vizinho mais próximo em SVD para permitir comparação.
As demais alternativas desta página são propostas futuras. Transporte ótimo (OT)
entre dois estágios fornece pesos
`P[i,j]` para fluxos entre células de origem `i` e de destino `j`, não pares
observados nem trajetórias individuais. Ajustar a projeção, marginais e custo
somente nas células de treino; repetir o ajuste separadamente na validação
interna. A reserva de avaliação permanece excluída.

## Interpretação e seleção de destinos

Para cada origem, normalize sua linha: `q(j|i) = P[i,j] / sum_j P[i,j]`.
Se uma linha tiver massa desprezível, marque-a como sem pareamento confiável,
em vez de dividir por quase zero. As marginais do OT determinam a massa total
de cada célula, enquanto `q(j|i)` determina o destino condicionado a uma origem.

- **Amostrar um destino por origem** segundo `q(j|i)` preserva, em expectativa,
  os vários destinos possíveis e mantém o formato atual do dataset. Uma amostra
  fixa tem variância alta; reamostrar a cada época pode reduzi-la ao longo do
  treino, mas dificulta reproduzir o mesmo pseudo-par.
- **Usar vários destinos ponderados** por origem diminui a variância do gradiente
  e retém multimodalidade. Aumenta custo e pode super-representar origens com
  mais destinos se seus pesos não somarem um por origem.
- **Usar a média ponderada** `sum_j q(j|i) x_j` dá alvo determinístico e barato
  após o cálculo, mas colapsa destinos divergentes para expressões intermediárias
  que podem não corresponder a uma célula real; tende a reduzir variância e
  diversidade. É um controle útil, não a recomendação principal.

Destinos podem ser reutilizados; sua frequência esperada deve seguir as
marginais de destino, não o número arbitrário de origens que os escolheram.
Para não confundir pareamento e distribuição real, a MMD do treino continua a
amostrar células futuras diretamente da população observada, com peso uniforme
se a avaliação tratar cada célula de forma uniforme. Se houver estimativas
biológicas de proliferação, morte ou viés amostral, elas podem informar massas
de origem/destino; sem essas estimativas, começar com marginais empíricas e
registrar o desvio das proporções celulares obtidas. Não force proporções iguais
entre estágios: o desenvolvimento altera abundâncias de linhagens.

O custo pode combinar distância na expressão projetada com informação biológica
confiável, mas a igualdade de tipo celular entre origem e destino não deve ser
uma restrição rígida: rótulos mudam com a diferenciação, podem ter granularidades
distintas e transições entre linhagens são parte do fenômeno a prever. Uma
penalidade suave só faria sentido após justificar a ontologia e examinar se
bloqueia transições conhecidas. Calcule um acoplamento separado por par de
estágios, usando seu `Δt` real; compare custos e regularização entre intervalos,
pois um salto de 0,25 dia e outro de 1 dia não devem receber automaticamente a
mesma expectativa de deslocamento.

## Primeiro experimento recomendado

Começar com OT entrópico no mesmo espaço SVD do vizinho atual, com uma amostra
de destino por origem a partir de `q(j|i)` e seed fixa. Manter todas as demais
opções, splits, genes, encoder e orçamento idênticos. Registrar a matriz de
pesos, massas de linhas/colunas, entropia por origem, distâncias, reutilização
de destinos e proporções celulares por transição. Ajustar a regularização
somente na validação interna. Comparar MSE de pseudo-pares e métricas de
distribuição em células não usadas no treino; em seguida repetir com seeds de
amostragem para medir variância. Se houver ganhos, testar reamostragem por época
e múltiplos destinos ponderados como experimentos separados. O alvo médio
ponderado serve como controle de possível perda de diversidade.

O cálculo da matriz custa memória proporcional a origens × destinos por par de
estágios. Teste primeiro em subconjuntos de treino: `--max-cells 128 --pairing ot`.
O treino salva `ot/summary.json` com tempo e memória estimada da matriz, além de
uma matriz `.npz` por transição e split. Consulte o [README](README.md) para a
comparação controlada. Reamostragem por época, vários destinos ponderados e alvo
médio ainda não foram implementados.
