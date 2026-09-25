# M3 — Geração de população conhecida

**Data:** 2026-09-24  
**Situação:** concluído com resultado negativo; gate não atendido.

## Método

E9.5 foi reconstruído usando as proporções e grupos reais K=50. Para cada célula sintética do grupo `k`:

```text
z_novo = média(E9.5, k) + resíduo_amostrado(E9.5, k)
```

O baseline repete somente `média(E9.5, k)`. Ambos usam a mesma amostragem multinomial de grupos e o decoder E011 congelado. Como E014 e E011 tinham tokenizações diferentes, os embeddings foram reextraídos com os 256 genes exatos do checkpoint do decoder; aplicar o decoder diretamente ao cache E014 foi corretamente rejeitado.

## Resultado

Resíduos preservaram a população no espaço latente: a variância ficou entre 0,979 e 1,015 do E9.5 real na sensibilidade de contagem. Após o decoder, porém, a razão de variância foi somente 0,0772, contra 0,0704 para centroides. Houve 506 expressões únicas com resíduos e apenas 45 com centroides, mas pseudobulk e RFF-MMD ficaram ligeiramente piores e covariância empatou.

Os dois `.h5ad` de 512 × 32.285 passaram o contrato estrutural. Não houve grupo sem resíduos nem fallback. A composição aproxima melhor o alvo conforme a contagem cresce: L1 0,431/0,206/0,118 para 128/512/2.048 células.

## Decisão

O mecanismo de centros + resíduos é válido como gerador latente simples, mas não supera centroides no pipeline completo por causa do colapso conhecido de E011. O gate M3 exige melhora no espaço de células e, portanto, não foi atendido. Não implementar nem treinar Transformer temporal antes de uma nova decisão do responsável e de uma intervenção controlada no decoder.
