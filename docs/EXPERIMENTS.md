# Registro de experimentos

## Como usar

1. Reserve o próximo ID antes da execução.
2. Declare hipótese e critério de decisão antes de observar resultados.
3. Registre inclusive resultados negativos.
4. Nunca sobrescreva resultados anteriores; acrescente uma nova entrada.

## Índice

| ID | Data | Marco | Hipótese | Configuração | Resultado principal | Decisão | Estado |
|---|---|---|---|---|---|---|---|
| E001 | pendente | M0 | O pipeline atual reproduz `copy_last` | pendente | pendente | pendente | PLANEJADO |
| E002 | pendente | M1 | Um decoder treinado sobre embeddings scGPT preserva a população melhor que repetir a média | checkpoint e dimensão pendentes de auditoria | pendente | pendente | PLANEJADO |
| E003 | pendente | M2 | K=100 equilibra detalhe e estabilidade | `K={50,100,200}` | pendente | pendente | PLANEJADO |
| E004 | pendente | M3 | Centros + resíduos empíricos preservam diversidade | pendente | pendente | pendente | PLANEJADO |

## Template detalhado

### EXXX — Título curto

**Data:**

**Marco:**

**Responsável/agente:**

**Hipótese:**

**Critério de decisão definido antes da execução:**

**Código:** commit, branch ou descrição do estado da árvore.

**Dados:** IDs de `DATA_POLICY.md`, splits e filtros.

**Configuração:**

```yaml
seed:
model:
training:
generation:
```

**Comandos:**

```bash
# preencher
```

**Métricas:**

| Métrica | Baseline | Experimento | Observação |
|---|---:|---:|---|
| pseudobulk |  |  |  |
| direção/DE |  |  |  |
| MMD |  |  |  |
| covariação/variograma |  |  |  |

**Artefatos:**

- checkpoint:
- métricas:
- figuras:
- predição:

**Resultado:**

**Limitações:**

**Decisão:** manter, rejeitar, repetir ou modificar.

**Próximo experimento:**
