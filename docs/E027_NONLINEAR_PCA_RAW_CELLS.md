# E027 — PCA não linear e células cruas

E027 comparou, no pseudo-holdout fixo E8.25→E8.5 de 512 células, uma PCA de kernel RBF aproximada por Nyström com duas entradas do Transformer: resumos K-means e células individuais. A variante individual foi treinada por perda populacional invariável à ordem, sem pares célula-a-célula.

O embedding usou 512 genes variáveis definidos apenas em D003–D010, 256 landmarks e 32 componentes. O retorno aos 32.285 genes comparou regressão ridge crua, soma de resíduo empírico de PCA e vizinho mais próximo.

O decoder cru colapsou a variância para 6,4–7,2%. Somar resíduo recuperou 92% da variância, mas a melhor variante de células cruas obteve DE 0,0952, direção 0,0576, MMD 0,12592 e variograma 0,007172. O E021, sob a mesma avaliação, obteve 0,0476, 0,1867, 0,07179 e 0,001332. Assim, houve ganho isolado de DE e regressão forte nas outras três métricas. O vizinho mais próximo com células cruas gerou somente 174 perfis únicos.

Todos os seis arquivos têm 512 × 32.285, `float32`, valores finitos e não negativos e ordem oficial dos genes. Dez testes focados passaram. Configuração, checkpoints, métricas e saídas estão em `outputs/population_transformer/E027/fixed_eval_seed42_n512/`.

Decisão: resultado negativo; não promover, não confirmar em 1.024 e preservar E021 como candidata atual.
