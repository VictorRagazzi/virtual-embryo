#!/usr/bin/env python3
"""
Virtual Embryo Challenge (NeurIPS 2026) — Task 1: Temporal

Prediz a distribuicao de expressao genica do coracao embrionario de
camundongo em E10.5, dados os estagios E8.5 e E9.5 (sem rastreamento
de celulas individuais ao longo do tempo).

Estrategia:
  Parte 1 — PCA conjunto + delta de centroide para cell types
            persistentes entre E8.5 e E9.5, aplicado sobre E9.5.
  Parte 2 — LLM (few-shot, few-shot com a transicao observada
            E8.5->E9.5) como prior biologico para decidir quais
            tipos NOVOS devem surgir em E10.5 (diferenciacao),
            em que proporcao, e de qual progenitor de E9.5 herdar
            o perfil de expressao.
  Parte 3 — Montagem final: persistentes (extrapolados) + tipos
            exclusivos de E9.5 (copiados) + tipos novos (amostrados
            do progenitor indicado), ate ~N_CELLS totais.

Uso:
    uv run python src/scripts/predict_e10_5.py \
        --e85 data/E85.h5ad \
        --e95 data/E95.h5ad \
        --out data/prediction_e10_5.h5ad
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from dotenv import load_dotenv

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
log = logging.getLogger("predict_e10_5")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

N_PCA_COMPONENTS = 50
TARGET_N_CELLS = 2500          # dentro da faixa exigida 1000-5118
MIN_SUBMISSION_CELLS = 1000
MAX_SUBMISSION_CELLS = 5118
CELLTYPE_COL = "celltype"
RANDOM_STATE = 0

DEFAULT_LLM_PROVIDER = "openrouter"
DEFAULT_LLM_MODEL = "deepseek/deepseek-v4-flash"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"


# ---------------------------------------------------------------------------
# Parte 0 — I/O e validacao
# ---------------------------------------------------------------------------

def load_data(
    path: str,
    label: str,
    max_cells: int | None = None,
    seed: int = RANDOM_STATE,
) -> ad.AnnData:
    log.info("Carregando %s de %s", label, path)
    if max_cells is None:
        adata = ad.read_h5ad(path)
    else:
        if max_cells <= 0:
            raise ValueError("--max-cells-per-stage deve ser positivo.")
        backed = ad.read_h5ad(path, backed="r")
        try:
            if max_cells >= backed.n_obs:
                adata = backed.to_memory()
            else:
                indices = np.sort(np.random.default_rng(seed).choice(backed.n_obs, max_cells, replace=False))
                adata = backed[indices].to_memory()
        finally:
            backed.file.close()
    if CELLTYPE_COL not in adata.obs.columns:
        raise ValueError(f"{label}: coluna '{CELLTYPE_COL}' ausente em .obs")
    log.info(
        "  %s: %d celulas x %d genes, %d tipos",
        label, adata.n_obs, adata.n_vars, adata.obs[CELLTYPE_COL].nunique(),
    )
    return adata


def densify(x) -> np.ndarray:
    """Converte matriz esparsa (se houver) para densa."""
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)



# ---------------------------------------------------------------------------
# Parte 1 — PCA conjunto + delta de centroide
# ---------------------------------------------------------------------------
 
@dataclass
class PcaContext:
    pca: PCA
    emb_e85: np.ndarray
    emb_e95: np.ndarray
    var_names: pd.Index
 
 
def fit_joint_pca(adata_e85: ad.AnnData, adata_e95: ad.AnnData,
                   n_comps: int = N_PCA_COMPONENTS) -> PcaContext:
    if not adata_e85.var_names.equals(adata_e95.var_names):
        raise ValueError("Ordem/identidade dos genes difere entre E8.5 e E9.5")
 
    x85 = densify(adata_e85.X)
    x95 = densify(adata_e95.X)
    x_concat = np.vstack([x85, x95])
 
    log.info("Ajustando PCA conjunto (%d componentes) em %d celulas", n_comps, x_concat.shape[0])
    pca = PCA(n_components=n_comps, random_state=RANDOM_STATE)
    emb_concat = pca.fit_transform(x_concat)
    log.info("  variancia explicada acumulada: %.3f", pca.explained_variance_ratio_.sum())
 
    emb_e85 = emb_concat[: x85.shape[0]]
    emb_e95 = emb_concat[x85.shape[0]:]
    return PcaContext(pca=pca, emb_e85=emb_e85, emb_e95=emb_e95, var_names=adata_e85.var_names)
 
 
def compute_centroids(embedding: np.ndarray, labels: pd.Series) -> dict[str, np.ndarray]:
    centroids = {}
    for ct in labels.unique():
        mask = (labels == ct).to_numpy()
        centroids[ct] = embedding[mask].mean(axis=0)
    return centroids
 
 
def compute_deltas(
    centroids_e85: dict[str, np.ndarray],
    centroids_e95: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    common = sorted(set(centroids_e85) & set(centroids_e95))
    log.info("Tipos persistentes (E8.5 -> E9.5): %d -> %s", len(common), common)
    return {ct: centroids_e95[ct] - centroids_e85[ct] for ct in common}
 
 
def extrapolate_persistent_types(
    adata_e95: ad.AnnData,
    ctx: PcaContext,
    deltas: dict[str, np.ndarray],
) -> ad.AnnData:
    """Aplica celulas_E10.5 = celulas_E9.5 + delta no espaco PCA, reprojeta."""
    labels = adata_e95.obs[CELLTYPE_COL]
    mask = labels.isin(deltas.keys()).to_numpy()
    if mask.sum() == 0:
        raise ValueError("Nenhuma celula de E9.5 pertence a um tipo persistente")
 
    emb_subset = ctx.emb_e95[mask].copy()
    labels_subset = labels[mask].to_numpy()
    for ct, delta in deltas.items():
        ct_mask = labels_subset == ct
        emb_subset[ct_mask] += delta
 
    x_recon = ctx.pca.inverse_transform(emb_subset)
    x_recon = np.clip(x_recon, 0, None)  # log1p nao pode ser negativo
 
    out = ad.AnnData(
        X=x_recon.astype(np.float32),
        obs=pd.DataFrame({CELLTYPE_COL: labels_subset}),
        var=pd.DataFrame(index=ctx.var_names),
    )
    log.info("Extrapolados %d celulas de tipos persistentes", out.n_obs)
    return out
 
 
def copy_e95_only_types(adata_e95: ad.AnnData, deltas: dict[str, np.ndarray]) -> ad.AnnData:
    """Tipos que so existem em E9.5 (sem par em E8.5): copiados sem shift."""
    labels = adata_e95.obs[CELLTYPE_COL]
    mask = (~labels.isin(deltas.keys())).to_numpy()
    only_types = sorted(labels[mask].unique())
    log.info("Tipos exclusivos de E9.5 (copiados sem shift): %s", only_types)
    out = adata_e95[mask].copy()
    out.obs = out.obs[[CELLTYPE_COL]]
    return out
 
 
# ---------------------------------------------------------------------------
# Parte 2 — LLM como prior biologico para tipos novos
# ---------------------------------------------------------------------------
 
NEW_CELLTYPES_SCHEMA_HINT = """
Responda APENAS com um objeto JSON (sem markdown, sem comentarios, sem texto
antes ou depois), exatamente no formato:
 
{
  "new_celltypes": [
    {
      "name": "string, nome do novo cell type",
      "proportion": 0.05,
      "progenitor_in_e95": "string, deve ser um dos tipos de E9.5 listados",
      "reasoning": "breve justificativa biologica (1-2 frases)"
    }
  ]
}
 
Regras:
- "proportion" e a fracao do total de celulas de E10.5 (soma de todas as
  entradas deve ser plausivel, tipicamente entre 0.05 e 0.3 no total).
- "progenitor_in_e95" DEVE ser exatamente um nome presente na lista de tipos
  de E9.5 fornecida.
- Nao inclua tipos que ja estao na lista de "tipos persistentes ja cobertos".
- Se nao houver evidencia biologica solida para um novo tipo, prefira nao
  inventa-lo.
"""
 
 
def build_llm_prompt(
    e85_counts: pd.Series,
    e95_counts: pd.Series,
    persistent_types: list[str],
    e95_only_types: list[str],
) -> tuple[str, str]:
    system_prompt = (
        "Voce e um especialista em biologia do desenvolvimento cardiaco de "
        "camundongo (embriologia cardiaca), auxiliando a prever quais tipos "
        "celulares novos emergem por diferenciacao entre os estagios E9.5 e "
        "E10.5 do coracao embrionario. Responda de forma estritamente "
        "factual, baseada em conhecimento estabelecido de linhagens "
        "cardiacas (segundo/primeiro campo cardiaco, crista neural cardiaca, "
        "proepicardio, endocardio, etc)."
    )
 
    e85_str = "\n".join(f"  - {ct}: {n} celulas" for ct, n in e85_counts.items())
    e95_str = "\n".join(f"  - {ct}: {n} celulas" for ct, n in e95_counts.items())
 
    appeared = sorted(set(e95_counts.index) - set(e85_counts.index))
    disappeared = sorted(set(e85_counts.index) - set(e95_counts.index))
 
    user_prompt = f"""
Transicao observada E8.5 -> E9.5 (unico par de estagios com dado real de
diferenciacao disponivel nesta especie/tecido):
 
Tipos em E8.5 ({e85_counts.sum()} celulas):
{e85_str}
 
Tipos em E9.5 ({e95_counts.sum()} celulas):
{e95_str}
 
Tipos que SURGIRAM entre E8.5 e E9.5 (nao existiam antes): {appeared}
Tipos que DESAPARECERAM entre E8.5 e E9.5 (nao ficam presentes depois): {disappeared}
 
Estado parcial de E10.5 ja extrapolado por PCA+delta (apenas tipos
persistentes entre E8.5 e E9.5, projetados adiante por analogia linear;
tipos exclusivos de E9.5 foram copiados sem alteracao):
 
Tipos persistentes ja cobertos (NAO reproponha estes): {persistent_types}
Tipos exclusivos de E9.5 ja copiados (podem ainda ser progenitores validos
de tipos novos, mas NAO devem ser reproponhos como "new_celltypes"): {e95_only_types}
 
Tarefa: usando conhecimento de embriologia cardiaca (padrao de
diferenciacao do miocardio ventricular/atrial, maturacao de camaras,
trabeculacao, formacao de valvas AV/OFT, contribuicao da crista neural
cardiaca, etc), determine quais tipos celulares NOVOS (que nao existiam
nem em E8.5 nem em E9.5) plausivelmente emergem em E10.5 por
diferenciacao a partir de um progenitor presente em E9.5, aplicando o
mesmo padrao de surgimento observado na transicao E8.5->E9.5.
 
{NEW_CELLTYPES_SCHEMA_HINT}
""".strip()
 
    return system_prompt, user_prompt
 
 
def _extract_json_block(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        return brace.group(0)
    return text
 
 
def call_llm(
    system_prompt: str,
    user_prompt: str,
    provider: str = DEFAULT_LLM_PROVIDER,
    model: str = DEFAULT_LLM_MODEL,
    api_key: str | None = None,
    max_tokens: int = 2000,
    temperature: float = 0.3,
    timeout: int = 60,
) -> str:
    """Funcao generica de chamada a LLM via stdlib (urllib), sem dependencia
    de SDK. Suporta providers com API OpenAI-compatible ("openrouter",
    "openai", etc) e a API nativa da Anthropic ("anthropic"). Trocar de
    provider = adicionar um branch aqui, o resto do pipeline nao muda.
    """
    if api_key is None:
        raise ValueError("api_key ausente (defina a variavel de ambiente correspondente)")
 
    if provider in ("openrouter", "openai"):
        url = {
            "openrouter": "https://openrouter.ai/api/v1/chat/completions",
            "openai": "https://api.openai.com/v1/chat/completions",
        }[provider]
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data_key_path = ("choices", 0, "message", "content")
 
    elif provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        data_key_path = ("content", 0, "text")
 
    else:
        raise ValueError(f"Provider desconhecido: {provider}")
 
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Erro HTTP {e.code} do provider {provider}: {err_body}") from e
 
    node = payload
    for key in data_key_path:
        node = node[key]
    return node
 
 
def query_llm_for_new_types(
    e85_counts: pd.Series,
    e95_counts: pd.Series,
    persistent_types: list[str],
    e95_only_types: list[str],
    valid_progenitors: set[str],
    provider: str,
    model: str,
    api_key: str,
    max_retries: int = 2,
) -> list[dict]:
    system_prompt, user_prompt = build_llm_prompt(
        e85_counts, e95_counts, persistent_types, e95_only_types
    )
 
    last_error = None
    for attempt in range(1, max_retries + 2):
        log.info("Chamando LLM para tipos novos (tentativa %d)", attempt)
        raw = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            provider=provider,
            model=model,
            api_key=api_key,
        )
        try:
            parsed = json.loads(_extract_json_block(raw))
            new_types = validate_llm_response(parsed, valid_progenitors)
            log.info("LLM propos %d tipo(s) novo(s): %s",
                      len(new_types), [t["name"] for t in new_types])
            return new_types
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            log.warning("Resposta invalida da LLM (%s), retentando...", e)
            user_prompt += (
                f"\n\nATENCAO: sua resposta anterior falhou na validacao "
                f"({e}). Responda APENAS com o JSON valido no formato pedido."
            )
 
    log.error("LLM nao retornou JSON valido apos %d tentativas (%s). "
              "Prosseguindo sem tipos novos.", max_retries + 1, last_error)
    return []
 
 
def validate_llm_response(parsed: dict, valid_progenitors: set[str]) -> list[dict]:
    if "new_celltypes" not in parsed or not isinstance(parsed["new_celltypes"], list):
        raise ValueError("chave 'new_celltypes' ausente ou nao e lista")
 
    validated = []
    for entry in parsed["new_celltypes"]:
        for key in ("name", "proportion", "progenitor_in_e95", "reasoning"):
            if key not in entry:
                raise ValueError(f"entrada sem campo obrigatorio '{key}': {entry}")
        if not isinstance(entry["proportion"], (int, float)) or not (0 < entry["proportion"] < 1):
            raise ValueError(f"'proportion' invalida em {entry}")
        if entry["progenitor_in_e95"] not in valid_progenitors:
            raise ValueError(
                f"'progenitor_in_e95'={entry['progenitor_in_e95']!r} nao esta "
                f"entre os tipos validos de E9.5"
            )
        validated.append(entry)
    return validated
 
 
# ---------------------------------------------------------------------------
# Parte 3 — Montagem final
# ---------------------------------------------------------------------------
 
def stratified_subsample(adata: ad.AnnData, n_target: int, rng: np.random.Generator) -> ad.AnnData:
    """Subamostra preservando as proporcoes relativas dos tipos presentes."""
    if adata.n_obs <= n_target:
        return adata.copy()
 
    labels = adata.obs[CELLTYPE_COL]
    counts = labels.value_counts()
    fracs = counts / counts.sum()
    keep_idx = []
    for ct, frac in fracs.items():
        n_ct = max(1, round(frac * n_target))
        ct_idx = np.where((labels == ct).to_numpy())[0]
        n_ct = min(n_ct, len(ct_idx))
        chosen = rng.choice(ct_idx, size=n_ct, replace=False)
        keep_idx.extend(chosen.tolist())
    keep_idx = np.array(keep_idx)
    if len(keep_idx) > n_target:
        keep_idx = rng.choice(keep_idx, size=n_target, replace=False)
    return adata[keep_idx].copy()
 
 
def sample_new_celltypes(
    new_types: list[dict],
    adata_e95: ad.AnnData,
    target_total: int,
    rng: np.random.Generator,
) -> ad.AnnData:
    labels = adata_e95.obs[CELLTYPE_COL]
    pieces = []
    for entry in new_types:
        n_cells = max(1, round(entry["proportion"] * target_total))
        progenitor = entry["progenitor_in_e95"]
        pool_idx = np.where((labels == progenitor).to_numpy())[0]
        replace = n_cells > len(pool_idx)
        chosen = rng.choice(pool_idx, size=n_cells, replace=replace)
 
        piece = adata_e95[chosen].copy()
        piece.obs[CELLTYPE_COL] = entry["name"]
        pieces.append(piece)
        log.info(
            "  novo tipo '%s': %d celulas amostradas de '%s' (replace=%s)",
            entry["name"], n_cells, progenitor, replace,
        )
 
    if not pieces:
        return ad.AnnData(
            X=np.empty((0, adata_e95.n_vars), dtype=np.float32),
            obs=pd.DataFrame({CELLTYPE_COL: []}),
            var=pd.DataFrame(index=adata_e95.var_names),
        )
    return ad.concat(pieces, join="outer")
 
 
def assemble_final(
    persistent: ad.AnnData,
    e95_only: ad.AnnData,
    new_types_adata: ad.AnnData,
    new_types: list[dict],
    target_total: int,
    rng: np.random.Generator,
) -> ad.AnnData:
    new_fraction = sum(e["proportion"] for e in new_types)
    new_fraction = min(new_fraction, 0.9)  # salvaguarda
    base_target = max(1, round(target_total * (1 - new_fraction)))
 
    base = ad.concat([persistent, e95_only], join="outer")
    base_sub = stratified_subsample(base, base_target, rng)
 
    parts = [base_sub]
    if new_types_adata.n_obs > 0:
        parts.append(new_types_adata)
    final = ad.concat(parts, join="outer")
 
    # ordem de genes precisa ser identica a do input
    final = final[:, persistent.var_names].copy()
 
    # ajuste fino para caber na faixa exigida pelos organizadores
    if final.n_obs > MAX_SUBMISSION_CELLS:
        idx = rng.choice(final.n_obs, size=MAX_SUBMISSION_CELLS, replace=False)
        final = final[idx].copy()
    elif final.n_obs < MIN_SUBMISSION_CELLS:
        log.warning(
            "Total final (%d) abaixo do minimo exigido (%d); "
            "considere reduzir a fracao reservada a tipos novos ou "
            "aumentar --target-cells.",
            final.n_obs, MIN_SUBMISSION_CELLS,
        )
 
    # embaralha ordem das celulas
    perm = rng.permutation(final.n_obs)
    final = final[perm].copy()
 
    log.info("Distribuicao final por tipo:\n%s",
              final.obs[CELLTYPE_COL].value_counts().to_string())
 
    # organizadores reclassificam: submissao nao deve ter a coluna celltype
    final.obs = pd.DataFrame(index=final.obs.index)
    return final
 
 
# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--e85", required=True, help="Caminho para E85.h5ad")
    p.add_argument("--e95", required=True, help="Caminho para E95.h5ad")
    p.add_argument("--out", required=True, help="Caminho de saida para prediction_e10_5.h5ad")
    p.add_argument("--n-comps", type=int, default=N_PCA_COMPONENTS)
    p.add_argument("--target-cells", type=int, default=TARGET_N_CELLS)
    p.add_argument("--max-cells-per-stage", type=int, default=None,
                   help="Limita cada estágio antes da densificação da PCA; use para auditoria segura.")
    # p.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER,
    #                 choices=["openrouter", "openai", "anthropic"])
    # p.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    # p.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV,
    #                 help="Nome da variavel de ambiente com a API key do provider")
    p.add_argument("--skip-llm", action="store_true",
                    help="Pula a Parte 2 (sem tipos novos), util para debug")
    p.add_argument("--seed", type=int, default=RANDOM_STATE)
    return p.parse_args()
 
def main() -> None:
    load_dotenv()

    args = parse_args()
    rng = np.random.default_rng(args.seed)

    llm_provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
    llm_model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)

    adata_e85 = load_data(args.e85, "E8.5", args.max_cells_per_stage, args.seed)
    adata_e95 = load_data(args.e95, "E9.5", args.max_cells_per_stage, args.seed + 1)

    # Parte 1 -----------------------------------------------------------
    ctx = fit_joint_pca(adata_e85, adata_e95, n_comps=args.n_comps)
    centroids_e85 = compute_centroids(
        ctx.emb_e85, adata_e85.obs[CELLTYPE_COL]
    )
    centroids_e95 = compute_centroids(
        ctx.emb_e95, adata_e95.obs[CELLTYPE_COL]
    )
    deltas = compute_deltas(centroids_e85, centroids_e95)

    persistent = extrapolate_persistent_types(
        adata_e95, ctx, deltas
    )
    e95_only = copy_e95_only_types(adata_e95, deltas)

    # Parte 2 -----------------------------------------------------------
    new_types: list[dict] = []

    if not args.skip_llm:
        api_key = None

        if llm_provider != "local":
            api_key = os.getenv(DEFAULT_API_KEY_ENV)

            if not api_key:
                log.warning(
                    "Variavel de ambiente %s nao definida; "
                    "pulando Parte 2.",
                    DEFAULT_API_KEY_ENV,
                )

        if llm_provider == "local" or api_key:
            log.info(
                "LLM provider: %s | model: %s",
                llm_provider,
                llm_model,
            )

            e85_counts = adata_e85.obs[CELLTYPE_COL].value_counts()
            e95_counts = adata_e95.obs[CELLTYPE_COL].value_counts()

            new_types = query_llm_for_new_types(
                e85_counts=e85_counts,
                e95_counts=e95_counts,
                persistent_types=sorted(deltas.keys()),
                e95_only_types=sorted(
                    e95_only.obs[CELLTYPE_COL].unique()
                ),
                valid_progenitors=set(
                    adata_e95.obs[CELLTYPE_COL].unique()
                ),
                provider=llm_provider,
                model=llm_model,
                api_key=api_key,
            )

    new_types_adata = sample_new_celltypes(
        new_types,
        adata_e95,
        args.target_cells,
        rng,
    )

    # Parte 3 -----------------------------------------------------------
    final = assemble_final(
        persistent=persistent,
        e95_only=e95_only,
        new_types_adata=new_types_adata,
        new_types=new_types,
        target_total=args.target_cells,
        rng=rng,
    )

    log.info(
        "Salvando predicao final: %d celulas x %d genes -> %s",
        final.n_obs,
        final.n_vars,
        args.out,
    )

    os.makedirs(
        os.path.dirname(args.out) or ".",
        exist_ok=True,
    )

    final.write_h5ad(args.out)
    log.info("Concluido.")

if __name__ == "__main__":
    sys.exit(main())
