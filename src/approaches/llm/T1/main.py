"""
Estagio 6 do pipeline LLM (Virtual Embryo Challenge - Task 1).

Le o transition_plan.json (saida do predict_transition.py) + os
cluster_cards.json (deltas de marcadores) + o E9.5 real, e materializa
uma tabela de celulas E10.5 com valores de expressao de verdade.

NENHUM valor numerico vem do LLM aqui -- essa etapa e' 100% deterministica:

  - Para cada tipo celular do plano, a "populacao base" e' amostrada das
    celulas REAIS de E9.5 daquele tipo (ou do "parent_celltype", se for
    uma linhagem nova / novel_lineage).
  - Tipos persistent/continuing_new: aplica delta_e8.5_to_e9.5 * trend_factor
    nos genes marcadores (extrapolacao linear escalada pelo LLM).
  - Tipos novel_lineage: aplica um deslocamento nos key_markers propostos
    pelo LLM, na direcao indicada, com magnitude proporcional ao desvio
    padrao do gene dentro do pool de origem (escala por nivel de
    confianca: high/medium/low -- ajustavel via CLI).
  - Adiciona ruido gaussiano pequeno em cima do delta aplicado (pra nao
    gerar clones identicos quando o pool real e' menor que o alvo).
  - Contagem de celulas por tipo via maior resto (largest remainder), pra
    bater exatamente --target-cells no total.

Uso:
    uv run python materialize_e10_5.py \
        --e95 data/E95.h5ad \
        --plan data/transition_plan.json \
        --cards data/cluster_cards.json \
        --target-cells 2500 \
        --out predictions/prediction_e10_5.h5ad
"""

import argparse
import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


def largest_remainder_counts(proportions: list[float], total: int) -> list[int]:
    raw = [p * total for p in proportions]
    floors = [int(np.floor(x)) for x in raw]
    remainder = total - sum(floors)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:remainder]:
        floors[i] += 1
    return floors


def dense_row(adata: ad.AnnData, idx: int) -> np.ndarray:
    row = adata.X[idx]
    return np.asarray(row.todense()).ravel() if sp.issparse(row) else np.asarray(row).ravel()


def gene_std_in_pool(adata: ad.AnnData, cell_indices: np.ndarray, gene_idx: int) -> float:
    sub = adata.X[cell_indices, gene_idx]
    vals = np.asarray(sub.todense()).ravel() if sp.issparse(sub) else np.asarray(sub).ravel()
    return float(vals.std()) if len(vals) > 0 else 0.0


def materialize(e95: ad.AnnData, plan: dict, cards_by_type: dict, target_cells: int,
                 conf_scale: dict, noise_frac: float, seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    gene_index = {g: i for i, g in enumerate(e95.var_names)}
    entries = plan["predicted_celltypes"]
    counts = largest_remainder_counts([e["target_proportion"] for e in entries], target_cells)

    rows = []
    obs_celltype = []
    obs_origin = []
    barcodes = []

    for entry, n_cells in zip(entries, counts):
        if n_cells <= 0:
            continue

        source_type = entry["parent_celltype"] if entry["origin"] == "novel_lineage" else entry["celltype"]
        pool_mask = (e95.obs["celltype"].astype(str) == source_type).values
        pool_idx = np.where(pool_mask)[0]
        if len(pool_idx) == 0:
            print(f"[aviso] sem celulas de origem ('{source_type}') para '{entry['celltype']}' -- pulando")
            continue

        replace = n_cells > len(pool_idx)
        sampled_idx = rng.choice(pool_idx, size=n_cells, replace=replace)

        # deltas a aplicar: gene -> (delta_base, escala_de_ruido)
        deltas: dict[str, float] = {}
        if entry["origin"] in ("persistent", "continuing_new"):
            card = cards_by_type.get(entry["celltype"])
            if card:
                for m in card["top_markers"]:
                    if m["delta_e8.5_to_e9.5"] is not None and m["gene"] in gene_index:
                        deltas[m["gene"]] = m["delta_e8.5_to_e9.5"] * entry["trend_factor"]
        else:  # novel_lineage
            for km in entry["key_markers"]:
                if km["gene"] not in gene_index:
                    continue
                sign = 1.0 if km["direction"] == "up" else -1.0
                std = gene_std_in_pool(e95, pool_idx, gene_index[km["gene"]])
                scale = conf_scale.get(km["confidence"], 0.5)
                magnitude = std * scale if std > 0 else 0.5 * scale
                deltas[km["gene"]] = sign * magnitude

        for ci in sampled_idx:
            vec = dense_row(e95, ci).copy()
            for gene, delta in deltas.items():
                gi = gene_index[gene]
                noise = rng.normal(0, abs(delta) * noise_frac) if delta != 0 else 0.0
                vec[gi] = max(0.0, vec[gi] + delta + noise)
            rows.append(vec)
            obs_celltype.append(entry["celltype"])
            obs_origin.append(entry["origin"])
            barcodes.append(f"E10.5-SYNTH-{len(barcodes):06d}-1")

    X = sp.csr_matrix(np.vstack(rows))
    obs = pd.DataFrame({"celltype": obs_celltype, "origin": obs_origin}, index=barcodes)
    out = ad.AnnData(X=X, obs=obs, var=e95.var.copy())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--e95", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--cards", required=True)
    ap.add_argument("--target-cells", type=int, default=2500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf-scale-high", type=float, default=1.5)
    ap.add_argument("--conf-scale-medium", type=float, default=0.8)
    ap.add_argument("--conf-scale-low", type=float, default=0.3)
    ap.add_argument("--noise-frac", type=float, default=0.15,
                     help="ruido gaussiano = noise_frac * |delta aplicado|, por celula")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    e95 = ad.read_h5ad(args.e95)
    with open(args.plan) as f:
        plan = json.load(f)
    with open(args.cards) as f:
        cards_by_type = {c["celltype"]: c for c in json.load(f)}

    conf_scale = {"high": args.conf_scale_high, "medium": args.conf_scale_medium, "low": args.conf_scale_low}
    out = materialize(e95, plan, cards_by_type, args.target_cells, conf_scale, args.noise_frac, args.seed)
    out.write_h5ad(args.out)

    print(f"{out.n_obs} celulas escritas em {args.out}")
    print(out.obs["celltype"].value_counts())


if __name__ == "__main__":
    main()