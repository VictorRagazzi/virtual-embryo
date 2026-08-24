"""
predict_e10_5.py
================
Prediz a distribuição de expressão gênica de E10.5 a partir de E8.5 e E9.5
usando Optimal Transport (Moscot) para aprender a dinâmica celular.

Estratégia:
  1. Moscot OT: ajusta T entre E8.5 → E9.5 no espaço PCA conjunto
  2. Extrapolação: cada célula E9.5 recebe um delta individual calculado
     via T (origem ponderada em E8.5), depois encontramos o vizinho mais
     próximo real em E9.5 (sem reconstrução de genes — CSS não sofre)
  3. LLM prior: consulta LLM para sugerir tipos celulares novos em E10.5
     que devem emergir por diferenciação, com proporções e progenitores
  4. Montagem final: combina células extrapoladas + novos tipos + amostragem

Uso:
  uv run python src/scripts/predict_e10_5.py \
      --e85 data/E85.h5ad \
      --e95 data/E95.h5ad \
      --out data/prediction_e10_5.h5ad
"""

import argparse
import json
import os
import urllib.request
import urllib.error
import warnings
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from dotenv import load_dotenv
from sklearn.neighbors import NearestNeighbors

from src.scripts.predict_e10_5_PCA import DEFAULT_API_KEY_ENV

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Configuração
# ─────────────────────────────────────────────────────────────────────────────

N_PCA = 100          # componentes PCA para o espaço de OT
N_TARGET = 2500     # número de células no output final
N_KNN = 5           # vizinhos para lookup final (diversidade)
OT_EPSILON = 0.05   # regularização entrópica do OT (mais alto = mais suave)
OT_MAX_ITER = 2000  # iterações Sinkhorn
    

# ─────────────────────────────────────────────────────────────────────────────
# Utilitários
# ─────────────────────────────────────────────────────────────────────────────

def to_dense(X) -> np.ndarray:
    """Converte sparse ou denso para ndarray float32."""
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


def call_llm_api(
    messages: list[dict],
    system: str,
    provider: str = "openrouter",
    model: Optional[str] = None,
    max_tokens: int = 10000,
) -> str:
    """
    Chamada genérica a APIs de LLM.

    Suporta:
      - "anthropic"   → api.anthropic.com  (chave: ANTHROPIC_API_KEY)
      - "openrouter"  → openrouter.ai       (chave: OPENROUTER_API_KEY)
      - "openai"      → api.openai.com      (chave: OPENAI_API_KEY)

    Para trocar de provedor basta alterar --llm-provider na linha de comando
    ou a variável de ambiente LLM_PROVIDER.
    """
    provider = provider.lower()

    if provider == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        url = "https://api.anthropic.com/v1/messages"
        default_model = "claude-sonnet-4-6"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model or default_model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }

    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        url = "https://openrouter.ai/api/v1/chat/completions"
        default_model = "deepseek/deepseek-v4-flash"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        # OpenRouter usa formato OpenAI — injetar system como primeiro msg
        openai_messages = [{"role": "system", "content": system}] + messages
        payload = {
            "model": model or default_model,
            "max_tokens": max_tokens,
            "messages": openai_messages,
        }

    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        url = "https://api.openai.com/v1/chat/completions"
        default_model = "gpt-4o"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        openai_messages = [{"role": "system", "content": system}] + messages
        payload = {
            "model": model or default_model,
            "max_tokens": max_tokens,
            "messages": openai_messages,
        }

    else:
        raise ValueError(f"Provedor desconhecido: {provider!r}. Use 'anthropic', 'openrouter' ou 'openai'.")


    if not api_key:
        print(f"  [LLM] AVISO: chave de API para '{provider}' não encontrada. Pulando LLM prior.")
        return ""

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")

    print(f" RESPOSTA: {req}")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  [LLM] HTTP {e.code}: {body[:300]}")
        return ""
    except Exception as e:
        print(f"  [LLM] Erro: {e}")
        return ""

    # Extrair texto dependendo do formato de resposta
    if provider == "anthropic":
        return result.get("content", [{}])[0].get("text", "")
    else:
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")


# ─────────────────────────────────────────────────────────────────────────────
# Parte 1 — Optimal Transport (Moscot)
# ─────────────────────────────────────────────────────────────────────────────

def run_optimal_transport(
    adata_e85: ad.AnnData,
    adata_e95: ad.AnnData,
    n_pca: int = N_PCA,
    epsilon: float = OT_EPSILON,
    max_iter: int = OT_MAX_ITER,
):
    """
    Ajusta o plano de transporte ótimo entre E8.5 e E9.5.

    Retorna:
        T        : matriz de transporte (n_e85, n_e95) como ndarray
        pca_model: objeto PCA do scanpy embutido no adata concatenado
        adata_cat: AnnData concatenado com X_pca calculado
    """
    from moscot.problems import TemporalProblem

    print("[OT] Concatenando E8.5 + E9.5...")
    adata_cat = ad.concat(
        [adata_e85, adata_e95],
        label="stage",
        keys=["e85", "e95"],
    )
    adata_cat.obs_names_make_unique()
    adata_cat.obs["time_float"] = (
        adata_cat.obs["stage"].map({"e85": 8.5, "e95": 9.5}).astype(float)
    )

    print(f"[OT] Calculando PCA conjunto ({n_pca} componentes)...")
    sc.pp.pca(adata_cat, n_comps=n_pca)

    print("[OT] Preparando TemporalProblem...")
    tp = TemporalProblem(adata_cat)
    tp = tp.prepare(
        time_key="time_float",
        joint_attr="X_pca",   # usa o PCA conjunto como espaço de custo
    )

    print(f"[OT] Resolvendo Sinkhorn (ε={epsilon}, max_iter={max_iter})...")
    tp = tp.solve(
        epsilon=epsilon,
        scale_cost="mean",
        max_iterations=max_iter,
    )

    T = np.array(tp.solutions[(8.5, 9.5)].transport_matrix)  # (n_e85, n_e95)
    print(f"[OT] T shape: {T.shape}, sum: {T.sum():.4f}")

    return T, adata_cat


def extrapolate_e105_via_ot(
    T: np.ndarray,
    adata_cat: ad.AnnData,
    n_target: int = N_TARGET,
    n_knn: int = N_KNN,
) -> ad.AnnData:
    """
    Extrapola E10.5 a partir do plano de transporte OT.

    Lógica:
      1. Para cada célula j em E9.5, sua 'origem ponderada' em E8.5 é:
            x_origin_j = sum_i  T_col_norm[i,j] * X_e85_pca[i]
         (onde T_col_norm é T normalizado por coluna)
      2. Delta individual: delta_j = X_e95_pca[j] - x_origin_j
      3. Posição extrapolada: X_extrap_j = X_e95_pca[j] + delta_j
      4. kNN lookup: para cada X_extrap_j, encontrar as k células mais
         próximas em E9.5 e amostrar uma — sem reconstrução de genes.

    Isso garante que todas as células no output são células reais de E9.5,
    preservando a biologia e evitando o round-trip PCA→genes que destruía o CSS.
    """
    idx_e85 = adata_cat.obs["stage"] == "e85"
    idx_e95 = adata_cat.obs["stage"] == "e95"

    X_e85_pca = adata_cat[idx_e85].obsm["X_pca"]  # (n_e85, n_pca)
    X_e95_pca = adata_cat[idx_e95].obsm["X_pca"]  # (n_e95, n_pca)

    print("[OT] Calculando deltas individuais via T...")
    # Normaliza T por coluna: cada coluna j = distribuição sobre E8.5 da célula j de E9.5
    T_col_sum = T.sum(axis=0, keepdims=True)  # (1, n_e95)
    T_col_norm = T / np.maximum(T_col_sum, 1e-12)  # (n_e85, n_e95)

    # Origem ponderada de cada célula E9.5 no espaço PCA
    x_origin_pca = T_col_norm.T @ X_e85_pca  # (n_e95, n_pca)

    # Delta: deslocamento que cada célula fez de E8.5 → E9.5
    delta_pca = X_e95_pca - x_origin_pca      # (n_e95, n_pca)

    # Extrapolação: aplicar o mesmo delta mais uma vez
    X_extrap_pca = X_e95_pca + delta_pca      # (n_e95, n_pca)

    print(f"[OT] kNN lookup (k={n_knn}) para encontrar células reais E9.5...")
    nn = NearestNeighbors(n_neighbors=n_knn, metric="euclidean", n_jobs=-1)
    nn.fit(X_e95_pca)
    distances, indices = nn.kneighbors(X_extrap_pca)  # (n_e95, k)

    # Para cada posição extrapolada, amostrar 1 dos k vizinhos
    # Pesos = 1/distância (células mais próximas são mais prováveis)
    rng = np.random.default_rng(42)
    chosen_local = []
    for i in range(len(X_extrap_pca)):
        d = distances[i]
        # Se vizinho colapsado (distância 0), pegar o primeiro
        if d.min() < 1e-9:
            chosen_local.append(indices[i, 0])
        else:
            w = 1.0 / d
            w /= w.sum()
            chosen_local.append(rng.choice(indices[i], p=w))

    chosen_local = np.array(chosen_local)  # índices locais dentro de E9.5

    # Amostrar n_target células (com reposição se necessário)
    sample_idx = rng.choice(len(chosen_local), size=n_target, replace=(n_target > len(chosen_local)))
    final_local_idx = chosen_local[sample_idx]

    # Recuperar as células E9.5 originais (expressão real, sem reconstrução)
    adata_e95_full = adata_cat[idx_e95]
    adata_pred = adata_e95_full[final_local_idx].copy()

    print(f"[OT] Extrapolação concluída: {adata_pred.n_obs} células")
    return adata_pred


# ─────────────────────────────────────────────────────────────────────────────
# Parte 2 — LLM prior para tipos celulares novos
# ─────────────────────────────────────────────────────────────────────────────

def query_llm_for_new_celltypes(
    celltypes_e85: dict,
    celltypes_e95: dict,
    provider: str = "anthropic",
    model: Optional[str] = None,
) -> list[dict]:
    """
    Consulta LLM para prever quais tipos celulares novos devem emergir em E10.5.

    O few-shot mostra o que aconteceu de E8.5 → E9.5 (tipos que sumiram,
    apareceram, proporções) e pede a mesma análise para E9.5 → E10.5.
    """
    # Calcular proporções
    total_e85 = sum(celltypes_e85.values())
    total_e95 = sum(celltypes_e95.values())

    e85_str = "\n".join(
        f"  - {ct}: {n} células ({100*n/total_e85:.1f}%)"
        for ct, n in sorted(celltypes_e85.items(), key=lambda x: -x[1])
    )
    e95_str = "\n".join(
        f"  - {ct}: {n} células ({100*n/total_e95:.1f}%)"
        for ct, n in sorted(celltypes_e95.items(), key=lambda x: -x[1])
    )

    only_e85 = set(celltypes_e85) - set(celltypes_e95)
    only_e95 = set(celltypes_e95) - set(celltypes_e85)
    common = set(celltypes_e85) & set(celltypes_e95)

    system_prompt = """Você é um especialista em embriologia cardíaca de camundongo.
Seu papel é usar seu conhecimento de biologia do desenvolvimento para prever
quais tipos celulares devem emergir durante o desenvolvimento cardíaco embrionário.
Você DEVE responder APENAS com JSON válido, sem texto antes ou depois, sem blocos markdown."""

    user_prompt = f"""## Contexto
Estamos prevendo a distribuição celular do coração embrionário de camundongo em E10.5.
Temos observações reais de E8.5 e E9.5 (de animais diferentes — sem tracking individual).

## Exemplo observado: transição E8.5 → E9.5

**E8.5** ({total_e85} células, 18 tipos):
{e85_str}

**E9.5** ({total_e95} células, 21 tipos):
{e95_str}

**O que aconteceu:**
- Tipos que desapareceram entre E8.5 e E9.5: {', '.join(only_e85) or 'nenhum'}
- Tipos novos que apareceram em E9.5: {', '.join(only_e95) or 'nenhum'}
- Tipos persistentes: {', '.join(sorted(common))}

## Tarefa
Considerando o desenvolvimento cardíaco de E9.5 → E10.5:

Em E10.5, o coração de camundongo passa por eventos críticos:
- Compactação do miocárdio ventricular
- Septação do outflow tract (OFT)
- Maturação do endocárdio e epicárdio
- Expansão de subtipos de cardiomiócitos

Quais tipos celulares NOVOS (não presentes em E9.5) devem emergir em E10.5?
Para cada tipo novo, indique:
1. O tipo celular progenitor mais adequado em E9.5 para herdar o perfil base
2. A proporção estimada do total de células de E10.5

IMPORTANTE:
- A soma das proporções de tipos novos deve ser ≤ 0.35 (o restante são tipos E9.5 que persistem)
- Inclua apenas tipos com evidência biológica clara para E10.5
- Use nomenclatura padrão de scRNA-seq cardíaco de camundongo

Responda APENAS com este JSON (sem markdown, sem texto extra):
{{
  "new_celltypes": [
    {{
      "name": "nome_do_tipo",
      "proportion": 0.XX,
      "progenitor_in_e95": "tipo_progenitor_em_e95",
      "reasoning": "justificativa biológica em 1-2 frases"
    }}
  ]
}}"""

    print("[LLM] Consultando LLM para prior de tipos celulares novos...")
    response_text = call_llm_api(
        messages=[{"role": "user", "content": user_prompt}],
        system=system_prompt,
        provider=provider,
        model=model,
    )

    if not response_text:
        print("[LLM] Sem resposta — usando prior vazio.")
        return []

    # Limpar markdown caso venha com backticks
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
    if cleaned.endswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[:-1])
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        new_cts = data.get("new_celltypes", [])
        print(f"[LLM] {len(new_cts)} tipos novos sugeridos:")
        for ct in new_cts:
            print(f"  → {ct['name']} ({ct['proportion']*100:.1f}%) "
                  f"← {ct['progenitor_in_e95']} | {ct['reasoning']}")
        return new_cts
    except json.JSONDecodeError as e:
        print(f"[LLM] Erro ao parsear JSON: {e}")
        print(f"[LLM] Resposta recebida:\n{response_text[:500]}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Parte 3 — Montagem final
# ─────────────────────────────────────────────────────────────────────────────

def add_new_celltypes(
    adata_base: ad.AnnData,
    adata_e95: ad.AnnData,
    new_celltypes: list[dict],
    n_total: int = N_TARGET,
) -> ad.AnnData:
    """
    Adiciona células de tipos novos sugeridos pela LLM ao pool base.

    Para cada tipo novo:
      - Seleciona células do progenitor indicado em E9.5
      - Amostra na proporção pedida do total final
      - Adiciona ao AnnData base

    Depois renormaliza para n_total células.
    """
    if not new_celltypes:
        return adata_base

    rng = np.random.default_rng(123)
    parts = [adata_base]
    total_new_prop = sum(ct.get("proportion", 0) for ct in new_celltypes)

    # Reduzir o pool base proporcionalmente para abrir espaço para novos tipos
    base_prop = 1.0 - total_new_prop
    n_base = max(100, int(n_total * base_prop))
    base_idx = rng.choice(adata_base.n_obs, size=n_base, replace=(n_base > adata_base.n_obs))
    parts = [adata_base[base_idx].copy()]

    available_e95_types = set(adata_e95.obs["celltype"].unique())

    for ct_info in new_celltypes:
        prog = ct_info.get("progenitor_in_e95", "")
        prop = ct_info.get("proportion", 0.0)
        name = ct_info.get("name", "unknown")
        n_new = max(10, int(n_total * prop))

        if prog not in available_e95_types:
            # Tentar match parcial
            matches = [t for t in available_e95_types if prog.lower() in t.lower()]
            if matches:
                prog = matches[0]
                print(f"  [Assembly] '{ct_info['progenitor_in_e95']}' → match parcial '{prog}'")
            else:
                print(f"  [Assembly] Progenitor '{prog}' não encontrado em E9.5, pulando '{name}'")
                continue

        mask = adata_e95.obs["celltype"] == prog
        pool = adata_e95[mask]
        idx = rng.choice(pool.n_obs, size=n_new, replace=(n_new > pool.n_obs))
        new_part = pool[idx].copy()
        parts.append(new_part)
        print(f"  [Assembly] {name}: {n_new} células (progenitor: {prog})")

    print("[Assembly] Concatenando partes...")
    adata_final = ad.concat(parts, merge="same")
    adata_final.obs_names_make_unique()

    # Renormalizar para exatamente n_total células
    if adata_final.n_obs > n_total:
        keep = rng.choice(adata_final.n_obs, size=n_total, replace=False)
        adata_final = adata_final[keep].copy()
    elif adata_final.n_obs < n_total:
        extra = rng.choice(adata_final.n_obs, size=n_total - adata_final.n_obs, replace=True)
        adata_final = ad.concat([adata_final, adata_final[extra]], merge="same")
        adata_final.obs_names_make_unique()

    print(f"[Assembly] Total final: {adata_final.n_obs} células")
    return adata_final


def prepare_submission(adata: ad.AnnData, adata_e95_ref: ad.AnnData) -> ad.AnnData:
    """
    Prepara o AnnData para submissão:
      - Remove coluna celltype do obs (organizadores reclassificam)
      - Garante que os genes estão na mesma ordem que E9.5
      - Mantém apenas X (sem obsm, obsp, uns extras)
    """
    # Garantir ordem de genes consistente
    shared_genes = list(adata_e95_ref.var_names)
    if list(adata.var_names) != shared_genes:
        adata = adata[:, shared_genes].copy()

    # Criar AnnData limpo só com X
    X_final = to_dense(adata.X)
    np.clip(X_final, 0, None, out=X_final)  # log1p não pode ser negativo

    adata_out = ad.AnnData(
        X=X_final,
        var=adata.var[[]].copy(),  # mantém var_names, sem colunas extras
    )
    adata_out.obs_names = [f"cell_{i}" for i in range(adata_out.n_obs)]

    return adata_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    api_key = os.getenv(DEFAULT_API_KEY_ENV)

    parser = argparse.ArgumentParser(description="Prediz E10.5 via Optimal Transport + LLM prior")
    parser.add_argument("--e85", required=True, help="Caminho para E85.h5ad")
    parser.add_argument("--e95", required=True, help="Caminho para E95.h5ad")
    parser.add_argument("--out", required=True, help="Caminho de saída .h5ad")
    parser.add_argument("--n-cells", type=int, default=N_TARGET,
                        help=f"Número de células no output (padrão: {N_TARGET})")
    parser.add_argument("--n-pca", type=int, default=N_PCA,
                        help=f"Componentes PCA (padrão: {N_PCA})")
    parser.add_argument("--epsilon", type=float, default=OT_EPSILON,
                        help=f"Regularização OT epsilon (padrão: {OT_EPSILON})")
    parser.add_argument("--n-knn", type=int, default=N_KNN,
                        help=f"Vizinhos para lookup kNN (padrão: {N_KNN})")
    parser.add_argument("--llm-provider", default=os.getenv("DEFAULT_LLM_PROVIDER", "openrouter"),
                        choices=["anthropic", "openrouter", "openai"],
                        help="Provedor de LLM (padrão: openrouter)")
    parser.add_argument("--llm-model", default=os.getenv("DEFAULT_LLM_MODEL", "deepseek/deepseek-v4-flash"),
                        help="Modelo específico do provedor (padrão: automático por provedor)")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Pular a consulta LLM (útil para debugging rápido)")
    args = parser.parse_args()

    # ── Carregar dados ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Carregando dados...")
    print(f"{'='*60}")
    adata_e85 = sc.read_h5ad(args.e85)
    adata_e95 = sc.read_h5ad(args.e95)
    print(f"E8.5: {adata_e85.n_obs} células × {adata_e85.n_vars} genes")
    print(f"E9.5: {adata_e95.n_obs} células × {adata_e95.n_vars} genes")

    celltypes_e85 = adata_e85.obs["celltype"].value_counts().to_dict()
    celltypes_e95 = adata_e95.obs["celltype"].value_counts().to_dict()

    # ── Parte 1: Optimal Transport ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PARTE 1: Optimal Transport (Moscot)")
    print(f"{'='*60}")
    T, adata_cat = run_optimal_transport(
        adata_e85, adata_e95,
        n_pca=args.n_pca,
        epsilon=args.epsilon,
    )

    print("\nExtrapolando E10.5...")
    adata_ot = extrapolate_e105_via_ot(
        T, adata_cat,
        n_target=args.n_cells,
        n_knn=args.n_knn,
    )

    # ── Parte 2: LLM prior ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PARTE 2: LLM prior para tipos celulares novos")
    print(f"{'='*60}")

    new_celltypes = []
    if not args.skip_llm:
        new_celltypes = query_llm_for_new_celltypes(
            celltypes_e85=celltypes_e85,
            celltypes_e95=celltypes_e95,
            provider=args.llm_provider,
            model=args.llm_model,
        )
    else:
        print("[LLM] Pulado (--skip-llm)")

    # ── Parte 3: Montagem final ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("PARTE 3: Montagem final")
    print(f"{'='*60}")
    adata_final = add_new_celltypes(
        adata_base=adata_ot,
        adata_e95=adata_e95,
        new_celltypes=new_celltypes,
        n_total=args.n_cells,
    )

    # ── Preparar e salvar submissão ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Preparando submissão...")
    print(f"{'='*60}")
    adata_submit = prepare_submission(adata_final, adata_e95)
    print(f"Shape final: {adata_submit.n_obs} células × {adata_submit.n_vars} genes")
    print(f"Colunas obs: {list(adata_submit.obs.columns)}")
    assert "celltype" not in adata_submit.obs.columns, "celltype não deve estar no output!"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    adata_submit.write_h5ad(args.out)
    print(f"\n✓ Salvo em: {args.out}")
    print(f"  {adata_submit.n_obs} células, {adata_submit.n_vars} genes")
    print(f"  X range: [{adata_submit.X.min():.3f}, {adata_submit.X.max():.3f}]")


if __name__ == "__main__":
    main()