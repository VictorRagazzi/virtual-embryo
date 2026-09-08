#!/usr/bin/env python
"""
predict_top_genes_llm.py

Abordagem LLM (genes) para o Virtual Embryo Challenge — Task 1 (Temporal).

Ideia geral
-----------
1. Parte do resultado do pré-processamento já existente (pre_processement.py):
   - genes removidos por baixa variância (repetidos sem alteração)
   - grupos de genes altamente correlacionados (só o representante é modelado;
     os demais membros do grupo são repetidos, a menos que --propagate-groups
     seja usado)
2. Nos genes representantes sobreviventes, calcula o delta real observado
   (E9.5 - E8.5) por gene, em nível populacional (média sobre as células).
3. Seleciona os top-N genes de maior |delta| (os que mais mudaram entre
   E8.5 e E9.5) e pede a uma LLM, em lotes, para classificar — usando como
   contexto um resumo de biologia do desenvolvimento (mouse_embryo.md) — se a
   tendência de cada gene deve, no próximo estágio (E9.5 -> E10.5):
       acelera | mantem_ritmo | desacelera | plato | reverte
   NÃO pedimos números à LLM, só uma categoria qualitativa. Cada categoria
   mapeia para um fator multiplicativo aplicado sobre o delta já observado.
4. Para os genes representantes fora do top-N: fator fixo = 1.0
   (extrapolação linear pura, E10.5_pred = E9.5 + delta).
5. Para genes excluídos (baixa variância + membros de grupos correlacionados):
   repete o valor de E9.5 (comportamento default).
6. Monta o h5ad final com todos os ~32k genes na mesma ordem do dataset de
   referência.

A predição é aplicada célula a célula: cada célula tem seu valor de E9.5
deslocado pelo delta populacional do gene (ponderado pelo fator), não um
valor médio único repetido pra todas as células — isso preserva a
heterogeneidade celular observada em E9.5.

Uso típico
----------
uv run python src/approaches/llm/T1/predict_top_genes_llm.py \
    --e85 data/E85.h5ad \
    --e95 data/E95.h5ad \
    --removed-genes data/preprocessing/joint_removed_genes.csv \
    --gene-groups data/preprocessing/joint_gene_groups.csv \
    --context-md src/approaches/llm/mouse_embryo.md \
    --out data/prediction_e10_5_llm_genes.h5ad \
    --top-n 75

Pré-requisitos: scanpy, anndata, pandas, numpy, scipy, openai, python-dotenv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("predict_top_genes_llm")


# ---------------------------------------------------------------------------
# Fatores de continuação de tendência
# ---------------------------------------------------------------------------
# fator multiplica o delta observado (E9.5 - E8.5) para extrapolar E10.5.
# fator = 1.0 -> extrapolação linear pura (mesma taxa de mudança continua)
LABEL_FACTORS: dict[str, float] = {
    "acelera": 1.5,        # tendência se intensifica
    "mantem_ritmo": 1.0,   # continua no mesmo ritmo (linear puro)
    "desacelera": 0.5,     # continua na mesma direção, mas mais devagar
    "plato": 0.15,         # praticamente estabiliza (perto de pico/vale biológico)
    "reverte": -0.5,       # tendência se inverte (ex.: gene transitório de EMT)
}
VALID_LABELS = set(LABEL_FACTORS.keys())


# ---------------------------------------------------------------------------
# Pré-processamento: carregar o que já foi filtrado/agrupado
# ---------------------------------------------------------------------------
@dataclass
class PreprocessingResult:
    removed_genes: set[str]
    group_members: dict[str, str]  # membro -> representante (exclui o próprio representante)
    representatives: set[str]      # todos os representantes de grupo (podem ser grupos de 1)


def load_preprocessing(removed_genes_csv: Path, gene_groups_csv: Path) -> PreprocessingResult:
    removed_genes: set[str] = set()
    if removed_genes_csv.exists():
        df = pd.read_csv(removed_genes_csv)
        # aceita a primeira coluna como nome do gene, independente do header exato
        gene_col = df.columns[0]
        removed_genes = set(df[gene_col].astype(str))
        log.info("Genes removidos por baixa variância: %d", len(removed_genes))
    else:
        log.warning("Arquivo de genes removidos não encontrado em %s (seguindo sem filtro)", removed_genes_csv)

    group_members: dict[str, str] = {}
    representatives: set[str] = set()
    if gene_groups_csv.exists():
        df = pd.read_csv(gene_groups_csv)
        # espera colunas tipo: representative, members (string separada por ; ou ,)
        rep_col = "representative" if "representative" in df.columns else df.columns[0]
        members_col = "members" if "members" in df.columns else df.columns[1]
        for _, row in df.iterrows():
            rep = str(row[rep_col])
            representatives.add(rep)
            members_raw = str(row[members_col])
            members = re.split(r"[;,]\s*", members_raw) if members_raw and members_raw != "nan" else []
            for m in members:
                m = m.strip()
                if m and m != rep:
                    group_members[m] = rep
        log.info("Grupos de genes correlacionados: %d representantes, %d membros redirecionados",
                  len(representatives), len(group_members))
    else:
        log.warning("Arquivo de grupos de genes não encontrado em %s (seguindo sem agrupamento)", gene_groups_csv)

    return PreprocessingResult(removed_genes=removed_genes, group_members=group_members, representatives=representatives)


# ---------------------------------------------------------------------------
# Cálculo de delta populacional por gene
# ---------------------------------------------------------------------------
def gene_population_means(adata: ad.AnnData) -> np.ndarray:
    """Média de expressão por gene (log1p-space), assumindo adata.X esparso."""
    x = adata.X
    if sparse.issparse(x):
        return np.asarray(x.mean(axis=0)).ravel()
    return np.asarray(x.mean(axis=0)).ravel()


def compute_deltas(e85: ad.AnnData, e95: ad.AnnData, genes: list[str]) -> pd.Series:
    """Delta médio populacional (E9.5 - E8.5) para os genes fornecidos."""
    idx85 = e85.var_names.get_indexer(genes)
    idx95 = e95.var_names.get_indexer(genes)
    missing = [g for g, i in zip(genes, idx85) if i == -1] + [g for g, i in zip(genes, idx95) if i == -1]
    if missing:
        raise ValueError(f"Genes não encontrados em ambos os datasets: {missing[:10]}...")

    mean85_full = gene_population_means(e85)
    mean95_full = gene_population_means(e95)
    delta = mean95_full[idx95] - mean85_full[idx85]
    return pd.Series(delta, index=genes)


def magnitude_bucket(z: float) -> str:
    """Describes the delta in qualitative terms (for the prompt) rather than exact numbers."""
    if z > 1.5:
        return "increased strongly"
    if z > 0.3:
        return "increased slightly"
    if z > -0.3:
        return "nearly stable"
    if z > -1.5:
        return "decreased slightly"
    return "decreased strongly"


# ---------------------------------------------------------------------------
# LLM: seleção de contexto e chamadas em lote
# ---------------------------------------------------------------------------
def load_context_md(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        log.warning("mouse_embryo.md não encontrado em %s — seguindo sem contexto biológico", path)
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        log.info("Contexto truncado de %d para %d chars", len(text), max_chars)
        text = text[:max_chars]
    return text


def build_prompt(batch: list[tuple[str, str]], context_md: str) -> str:
    """batch: list of (gene, qualitative description of delta E8.5->E9.5)."""
    genes_block = "\n".join(f"- {gene}: {desc} from E8.5 to E9.5" for gene, desc in batch)
    return f"""You are an expert in mouse embryonic cardiac developmental biology.

Context (literature summary on embryonic heart development):
{context_md if context_md else "(no additional context provided)"}

For each gene below, we know how it changed from E8.5 to E9.5 (qualitative direction
only, not numerical values). Your task is to predict, based on cardiac developmental
biology knowledge, which of the 5 categories best describes what should happen to
this gene in the next transition, E9.5 -> E10.5:

- "acelera": the observed trend should intensify
- "mantem_ritmo": the trend should continue at the same pace
- "desacelera": the trend continues in the same direction, but slower
- "plato": the gene should stabilize (near a biological peak or trough)
- "reverte": the trend should reverse (e.g., transient differentiation gene)

Genes:
{genes_block}

Respond ONLY with JSON (no markdown, no extra text) in the format:
{{"predictions": [{{"gene": "<name>", "label": "<one of the 5 categories>", "rationale": "<1 sentence>"}}, ...]}}
"""


def call_llm(client, model: str, prompt: str, temperature: float = 0.2) -> list[dict]:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.error("Falha ao parsear resposta da LLM, conteúdo bruto:\n%s", raw)
        raise
    return parsed.get("predictions", [])


def make_llm_client(provider: str):
    from openai import OpenAI

    if provider == "openrouter":
        api_key = os.environ["OPENROUTER_API_KEY"]
        base_url = "https://openrouter.ai/api/v1"
    elif provider == "openai":
        api_key = os.environ["OPENAI_API_KEY"]
        base_url = None
    else:
        api_key = os.environ.get(f"{provider.upper()}_API_KEY", os.environ.get("LLM_API_KEY"))
        base_url = os.environ.get("LLM_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


def classify_top_genes(
    top_genes: list[str],
    deltas: pd.Series,
    context_md: str,
    client,
    model: str,
    batch_size: int,
) -> dict[str, str]:
    """Retorna gene -> label. Genes que falharem no parsing caem em 'mantem_ritmo'."""
    z = (deltas - deltas.mean()) / (deltas.std() + 1e-9)
    labels: dict[str, str] = {}

    batches = [top_genes[i:i + batch_size] for i in range(0, len(top_genes), batch_size)]
    for bi, batch_genes in enumerate(batches, 1):
        batch = [(g, magnitude_bucket(z[g])) for g in batch_genes]
        prompt = build_prompt(batch, context_md)
        log.info("Chamando LLM para lote %d/%d (%d genes)...", bi, len(batches), len(batch_genes))
        try:
            predictions = call_llm(client, model, prompt)
        except Exception as e:
            log.error("Erro na chamada da LLM para o lote %d: %s — usando 'mantem_ritmo' para todo o lote", bi, e)
            for g in batch_genes:
                labels[g] = "mantem_ritmo"
            continue

        seen = set()
        for pred in predictions:
            gene = pred.get("gene")
            label = pred.get("label")
            if gene not in batch_genes:
                continue
            if label not in VALID_LABELS:
                log.warning("Label inválido '%s' para gene %s — usando 'mantem_ritmo'", label, gene)
                label = "mantem_ritmo"
            labels[gene] = label
            seen.add(gene)

        for g in batch_genes:
            if g not in seen:
                log.warning("Gene %s não retornado pela LLM — usando 'mantem_ritmo'", g)
                labels[g] = "mantem_ritmo"

    return labels


# ---------------------------------------------------------------------------
# Montagem da matriz final
# ---------------------------------------------------------------------------
def assemble_prediction(
    e85: ad.AnnData,
    e95: ad.AnnData,
    pre: PreprocessingResult,
    top_genes: list[str],
    gene_labels: dict[str, str],
    clip_multiplier: float,
    propagate_groups: bool,
) -> ad.AnnData:
    all_genes = list(e95.var_names)
    n_cells, n_genes = e95.shape

    x95 = e95.X.tocsc() if sparse.issparse(e95.X) else sparse.csc_matrix(e95.X)
    x_pred = x95.copy().tolil()

    idx85 = {g: i for i, g in enumerate(e85.var_names)}
    idx95 = {g: i for i, g in enumerate(e95.var_names)}
    mean85 = gene_population_means(e85)
    mean95 = gene_population_means(e95)
    max_obs = np.maximum(
        np.asarray(e85.X.max(axis=0).todense()).ravel() if sparse.issparse(e85.X) else e85.X.max(axis=0),
        np.asarray(e95.X.max(axis=0).todense()).ravel() if sparse.issparse(e95.X) else e95.X.max(axis=0),
    )

    representatives = pre.representatives if pre.representatives else set(all_genes) - pre.group_members.keys()
    excluded = pre.removed_genes | set(pre.group_members.keys())

    n_top_applied, n_linear_applied, n_repeated = 0, 0, 0

    for gene in all_genes:
        col95 = idx95[gene]

        if gene in excluded and not (propagate_groups and gene in pre.group_members):
            # genes removidos por baixa variância, ou membros de grupo (default: repete E9.5)
            n_repeated += 1
            continue

        if gene not in idx85:
            # gene não existe em E8.5 (novo tipo/estado emergente) — sem delta possível, repete
            n_repeated += 1
            continue

        if propagate_groups and gene in pre.group_members:
            rep = pre.group_members[gene]
            if rep not in idx85 or rep not in idx95:
                n_repeated += 1
                continue
            factor = LABEL_FACTORS[gene_labels.get(rep, "mantem_ritmo")] if rep in top_genes else 1.0
            delta = mean95[idx95[rep]] - mean85[idx85[rep]]
        elif gene in top_genes:
            factor = LABEL_FACTORS[gene_labels[gene]]
            delta = mean95[col95] - mean85[idx85[gene]]
            n_top_applied += 1
        elif gene in representatives:
            factor = 1.0
            delta = mean95[col95] - mean85[idx85[gene]]
            n_linear_applied += 1
        else:
            n_repeated += 1
            continue

        shift = factor * delta
        col_vals = np.asarray(x95[:, col95].todense()).ravel() + shift
        upper = max(max_obs[col95] * clip_multiplier, 1e-6)
        col_vals = np.clip(col_vals, 0.0, upper)
        x_pred[:, col95] = col_vals.reshape(-1, 1)

    log.info(
        "Genes: %d via LLM (top-N), %d via extrapolação linear, %d repetidos sem alteração",
        n_top_applied, n_linear_applied, n_repeated,
    )

    pred_adata = ad.AnnData(
        X=x_pred.tocsr(),
        obs=e95.obs.copy(),
        var=e95.var.copy(),
    )
    if "X_umap.harmony.rna" in e95.obsm:
        pred_adata.obsm["X_umap.harmony.rna"] = e95.obsm["X_umap.harmony.rna"].copy()
    return pred_adata


def subsample_cells(adata: ad.AnnData, target_cells: int, seed: int) -> ad.AnnData:
    if adata.n_obs <= target_cells:
        return adata
    rng = np.random.default_rng(seed)
    idx = rng.choice(adata.n_obs, size=target_cells, replace=False)
    idx.sort()
    return adata[idx].copy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--e85", type=Path, default=Path("data/E85.h5ad"))
    p.add_argument("--e95", type=Path, default=Path("data/E95.h5ad"))
    p.add_argument("--removed-genes", type=Path, default=Path("data/preprocessing/joint_removed_genes.csv"))
    p.add_argument("--gene-groups", type=Path, default=Path("data/preprocessing/joint_gene_groups.csv"))
    p.add_argument("--context-md", type=Path, default=Path("src/approaches/llm/mouse_embryo.md"))
    p.add_argument("--out", type=Path, default=Path("data/prediction_e10_5_llm_genes.h5ad"))
    p.add_argument("--top-n", type=int, default=75, help="Quantidade de genes mais variáveis avaliados pela LLM (50-100 recomendado)")
    p.add_argument("--batch-size", type=int, default=15, help="Genes por chamada de LLM")
    p.add_argument("--target-cells", type=int, default=2500, help="Faixa aceita pela competição: 1000-5118")
    p.add_argument("--clip-multiplier", type=float, default=1.2, help="Teto de segurança: valor máx observado x este fator")
    p.add_argument("--propagate-groups", action="store_true", help="Em vez de repetir, escala membros de grupo proporcionalmente ao representante")
    p.add_argument("--skip-llm", action="store_true", help="Debug: usa 'mantem_ritmo' (linear puro) em vez de chamar a LLM")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    log.info("Carregando E8.5 e E9.5...")
    e85 = ad.read_h5ad(args.e85)
    e95 = ad.read_h5ad(args.e95)

    log.info("Carregando resultado do pré-processamento...")
    pre = load_preprocessing(args.removed_genes, args.gene_groups)

    representatives = pre.representatives if pre.representatives else (
        set(e95.var_names) - pre.removed_genes - pre.group_members.keys()
    )
    representatives = sorted(g for g in representatives if g in e85.var_names and g in e95.var_names)
    log.info("Genes representantes sobreviventes ao pré-processamento: %d", len(representatives))

    deltas = compute_deltas(e85, e95, representatives)
    top_genes = deltas.abs().sort_values(ascending=False).head(args.top_n).index.tolist()
    log.info("Top-%d genes selecionados por |delta| E8.5->E9.5", len(top_genes))

    if args.skip_llm:
        log.info("--skip-llm ativo: usando 'mantem_ritmo' para todos os top genes")
        gene_labels = {g: "mantem_ritmo" for g in top_genes}
    else:
        context_md = load_context_md(args.context_md)
        provider = os.environ.get("LLM_PROVIDER", "openrouter")
        model = os.environ.get("LLM_MODEL", "deepseek/deepseek-v4-flash")
        client = make_llm_client(provider)
        gene_labels = classify_top_genes(top_genes, deltas, context_md, client, model, args.batch_size)

    label_counts = pd.Series(gene_labels).value_counts()
    log.info("Distribuição de labels da LLM:\n%s", label_counts.to_string())

    pred_adata = assemble_prediction(
        e85, e95, pre, top_genes, gene_labels,
        clip_multiplier=args.clip_multiplier,
        propagate_groups=args.propagate_groups,
    )

    pred_adata = subsample_cells(pred_adata, args.target_cells, args.seed)
    log.info("Predição final: %d células x %d genes", pred_adata.n_obs, pred_adata.n_vars)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pred_adata.write_h5ad(args.out)
    log.info("Salvo em %s", args.out)


if __name__ == "__main__":
    sys.exit(main())