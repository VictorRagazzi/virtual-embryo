# M5 — Transformer temporal e predição exploratória

**Data:** 2026-09-24  
**Situação:** pipeline concluído; resultado temporal negativo; artefatos não submetíveis.

## Dados e split

A dinâmica usa exclusivamente D003–D011 em janelas de dois estágios. D011/E8.5 foi holdout em E017/E018 e não participou de normalização, treino ou seleção. O cache K=50 foi ajustado conjuntamente em E014/E015 e inclui o holdout e D001/D002; isso é vazamento estrutural conhecido. Para E019, o Transformer foi reajustado em todas as janelas conhecidas D003–D011 e recebeu os estados oficiais D001/E8.5 e D002/E9.5 para extrapolar E10.5. Nenhum arquivo E10.5 real foi usado.

## Modelo

Cada token contém tempo relativo ao alvo, proporção, centro CLS 512, dispersão CLS 512 e máscara de grupo vazio. Uma projeção linear alimenta um Transformer numérico de dimensão 128, duas camadas, quatro cabeças e dropout 0,05. As saídas usam softmax para proporções e softplus para dispersões. A loss soma MSE de proporções e MSE de centros/dispensões somente nos grupos observados do alvo. Seed 42 e algoritmos determinísticos foram ativados.

## Holdout E8.5

Copy-last obteve erro agregado 0,00372; extrapolação linear 0,16389; MLP 0,00941; Transformer 0,01976. A extrapolação linear foi melhor apenas em composição (L1 0,107). Portanto não há evidência de que atenção melhore esta tarefa com seis exemplos de treino; copy-last é a configuração recomendada pela pseudoavaliação.

## Geração E10.5

E019 gerou 512 latentes segundo proporções previstas, centros, dispersões e resíduos latentes E9.5. Foram exportadas três variantes: decoder cru, decoder + resíduos de expressão E9.5 do mesmo grupo e vizinho E9.5. Não houve fallback. A variante residual preservou razão de variância 0,882 contra referência E9.5, versus 0,041 do decoder cru, mas perdeu covariação. Esses diagnósticos não medem acurácia em E10.5.

Todos os `.h5ad` têm 32.285 genes na ordem oficial, `float32`, valores finitos/não negativos e faixa compatível com os dados. Permanecem `USO_APENAS_EXPLORATORIO_NAO_SUBMISSAO` devido ao checkpoint M001 e não atendem necessariamente a contagem oficial atual.

## Recomendação

Para o relatório acadêmico, apresentar o pipeline como completo e o Transformer como resultado negativo válido. A variante de expressão recomendada para inspeção exploratória é decoder + resíduo E9.5; vizinho E9.5 é somente baseline de memorização. Para submissão, resolver proveniência do encoder, reconfirmar regras e limites, validar com ferramenta oficial e congelar uma dinâmica sustentada pelo holdout.
