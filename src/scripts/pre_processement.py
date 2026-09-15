"""
pre_processement.py – Pré-processamento de expressão gênica para o Virtual Embryo Challenge.

Contrato T2.01 (docs/tasks/T2_01_PREPROCESSAMENTO.md):
  1. Estatísticas por gene calculadas sobre E8.5 + E9.5 juntos (nunca só E8.5).
  2. Cálculo por gene, em blocos e acumuladores float64, sem materializar a
     matriz conjunta densa:
       count, soma, soma de quadrados, nº de valores non-zero,
       variance = E[x²] − E[x]²  (clamp ≥ 0 por erro numérico).
  3. Filtro de variância separado do agrupamento por correlação
     (`--group-correlated` é opt-in: é caro e não necessário para o decoder).
  4. Amostragem explícita e estratificada por `celltype` dentro de cada
     estágio. Default de desenvolvimento: `--sample-cells 2000` por estágio.
     0 = todas as células (imprime aviso claro de memória).
  5. Artefatos em subdirectório identificado pela configuración; nunca
     sobrescribe silenciosamente una execução incompatível:
         kept_genes.txt      → genes mantidos na ordem canónica de var_names
         removed_genes.csv   → nome, variância, média, fração non-zero
         gene_stats.csv      → todos os genes + columna booleana `kept`
         manifest.json       → fingerprint completo da execução
  6. Manifest com: entradas, totais/amostrados por estágio, seed e estratégia,
     threshold, escala declarada, nº mantidos/removidos, SHA-256 de var_names
     e de obs_names usados por estágio, versão do formato.
  7. Escala obrigatoriamente explícita: `--input-scale {log1p,counts}`.
  8. Verifica que E8.5 e E9.5 têm exactamente os mesmos genes na mesma ordem.
  9. Falha se nenhun/todos os genes são mantidos (salvo teste sintético com
     `--allow-extreme-filter`).
  10. Não escreve cópias densas automáticas de E8.5/E9.5 filtrados; os
      consumidores devem aplicar `kept_genes.txt` ao cargar lotes.
      EXCEÇÃO OPT-IN: `--write-filtered-h5ad` escreve explicitamente
      `e85_filtered.h5ad` / `e95_filtered.h5ad` (painel filtrado + células
      amostradas) dentro do próprio `run_dir`, para quem precisa consumir os
      dados já prontos sem reimplementar o subset em outro script. Continua
      desligado por padrão: só materializa cópias densas quando pedido
      explicitamente, igual a `--group-correlated`.

Uso (smoke test de T2.01):

    uv run python src/scripts/pre_processement.py \\
        --e85 data/E85_sample.h5ad \\
        --e95 data/E95_sample.h5ad \\
        --sample-cells 500 \\
        --input-scale log1p \\
        --variance-threshold 0.01 \\
        --output-dir data/t2_smoke/preprocessing

Uso (idem, mas já escrevendo os .h5ad filtrados prontos para consumo):

    uv run python src/scripts/pre_processement.py \\
        --e85 data/E85_sample.h5ad \\
        --e95 data/E95_sample.h5ad \\
        --sample-cells 500 \\
        --input-scale log1p \\
        --variance-threshold 0.01 \\
        --output-dir data/t2_smoke/preprocessing \\
        --write-filtered-h5ad
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Iterable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse

# ─── Constantes ───────────────────────────────────────────────────────────────

MANIFEST_FORMAT_VERSION = 1
DEFAULT_VARIANCE_THRESHOLD = 0.01
DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_SAMPLE_CELLS = 2_000  # por estágio; 0 → todas (con aviso de memória)
DEFAULT_SEED = 0
DEFAULT_OUTPUT_DIR = Path("data/preprocessing")
BLOCK_CELLS = 512  # filas por bloco ao acumular estatísticas
ARTIFACT_PATHS = ("kept_genes.txt", "removed_genes.csv", "gene_stats.csv")
FILTERED_H5AD_PATHS = ("e85_filtered.h5ad", "e95_filtered.h5ad")


# ─── Hashes ───────────────────────────────────────────────────────────────────

def sha256_of_sequence(values: Iterable[object]) -> str:
    """SHA-256 de una sequência ordenada de nomes (genes ou células)."""
    h = hashlib.sha256()
    for v in values:
        h.update(str(v).encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    """SHA-256 (en bloques) do contenido de um arquivo."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ─── Carregamento e validação ─────────────────────────────────────────────────

def load_dataset(path: Path, label: str) -> ad.AnnData:
    """Carga un .h5ad normal (não backed) e imprime dimensiones."""
    print(f"\n📂  Carregando {label}: {path}")
    adata = sc.read_h5ad(path)
    print(f"    {adata.n_obs:,} células × {adata.n_vars:,} genes")
    return adata


def validate_same_gene_order(e85: ad.AnnData, e95: ad.AnnData) -> None:
    """T2.01-8: E8.5 e E9.5 devem ter exactamente os mesmos genes, na mesma ordem."""
    v85, v95 = list(e85.var_names), list(e95.var_names)
    if v85 == v95:
        return
    if len(v85) != len(v95):
        raise ValueError(
            f"E8.5 e E9.5 têm nº de genes diferente ({len(v85):,} vs {len(v95):,}). "
            "Devem compartir exactamente o mesmo painel de genes, na mesma ordem."
        )
    mismatches = [(i, v85[i], v95[i]) for i in range(len(v85)) if v85[i] != v95[i]]
    preview = "; ".join(f"{i}: {a} != {b}" for i, a, b in mismatches[:5])
    raise ValueError(
        f"E8.5 e E9.5 não compartilham a mesma ordem de genes "
        f"({len(mismatches):,} posições diferentes). Primeras divergências: {preview}."
    )


def check_scale_consistency(adata: ad.AnnData, declared_scale: str, label: str) -> float:
    """T2.01-7: detecta incoerencias graves entre `--input-scale` e os datos.

    NUNCA decide a escala por si mesmo (não há heurística `max <= 20 → log1p`
    silenciosa): a escala é declarada pelo usuário e só se emite un aviso se o
    máximo observado parece inconsistente com ela.
    """
    X = adata.X
    if issparse(X):
        maxv = float(X.data.max()) if X.nnz else 0.0
    else:
        maxv = float(np.max(X)) if X.size else 0.0
    if declared_scale == "log1p" and maxv > 30:
        print(f"    ⚠ [{label}] escala declarada 'log1p' mas o máximo observado "
              f"é {maxv:.3f} (suspeito).")
    if declared_scale == "counts" and maxv <= 1:
        print(f"    ⚠ [{label}] escala declarada 'counts' mas o máximo observado "
              f"é {maxv:.3f} (suspeito).")
    return maxv


# ─── Amostragem estratificada ─────────────────────────────────────────────────

def sample_cells(
    adata: ad.AnnData,
    n_requested: int,
    seed: int,
    label: str,
) -> tuple[ad.AnnData, str]:
    """T2.01-4: amostra `n_requested` células de um estágio.

    Estratificada por `celltype` (asignación por resto maior) e determinista em
    `seed`. Se `celltype` não existe em obs, cae a aleatoria simple (con aviso).
    `n_requested <= 0` ou `>= n_obs` devolve todas as células (con aviso claro
    de memória no caso 0).

    Retorna (annData amostrado, nome da estratégia).
    """
    observed = adata.n_obs
    if n_requested <= 0 or n_requested >= observed:
        if n_requested <= 0:
            print(f"    ⚠ [{label}] --sample-cells 0 → se usa TODAS as células "
                  f"({observed:,}). Com dados completos isto pode esgotar a memória.")
        return adata.copy(), "all-cells"

    rng = np.random.default_rng(seed)
    if "celltype" not in adata.obs.columns:
        print(f"    ⚠ [{label}] coluna 'celltype' ausente em obs; amostragem "
              f"aleatoria simple.")
        idx = rng.choice(observed, size=n_requested, replace=False)
        return adata[idx].copy(), "simple-random"

    groups = adata.obs["celltype"].astype(str).to_numpy()
    uniq, inv = np.unique(groups, return_inverse=True)
    counts = np.bincount(inv, minlength=len(uniq))

    # Asignación proporcional por resto maior (largest remainder method).
    frac = counts * (n_requested / observed)
    alloc = np.floor(frac).astype(int)
    remainder = n_requested - int(alloc.sum())
    frac_part = frac - alloc
    order = np.argsort(-frac_part, kind="stable")
    i = 0
    while remainder > 0:
        gi = order[i % len(order)]
        if alloc[gi] < counts[gi]:
            alloc[gi] += 1
            remainder -= 1
        i += 1
        if i > len(order) * 8:  # salvaguarda anti-loop infinito
            raise RuntimeError("amostragem estratificada não convergió.")

    idx_parts = []
    for gi in range(len(uniq)):
        k = alloc[gi]
        if k <= 0:
            continue
        members = np.where(inv == gi)[0]
        idx_parts.append(rng.choice(members, size=k, replace=False))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)  # ordem final determinista dado o seed
    return adata[idx].copy(), "stratified-celltype"


# ─── Estatísticas por gene (conjuntas E8.5 + E9.5) ───────────────────────────

def compute_gene_stats(
    e85: ad.AnnData,
    e95: ad.AnnData,
    block_cells: int = BLOCK_CELLS,
) -> pd.DataFrame:
    """T2.01-1/2: estatísticas por gene sobre E8.5 + E9.5 juntos.

    Percorre as duas matrices em bloques de `block_cells` filas, acumulando em
    float64:
        count, soma, soma de quadrados ← E[x²], nº de non-zero
    e calcula:
        mean       = soma / count
        variance   = E[x²] − E[x]²   (clamp ≥ 0 por erro numérico)
        pct_nonzero= nº non-zero / count

    NUNCA materializa a matriz conjunta densa (T2.01-2).
    """
    n_genes = e85.n_vars
    gene_names = np.asarray(e85.var_names, dtype=str)
    n_cells = 0
    n_nonzero = np.zeros(n_genes, dtype=np.int64)
    x_sum = np.zeros(n_genes, dtype=np.float64)
    x_sum_sq = np.zeros(n_genes, dtype=np.float64)

    def _accumulate(adata: ad.AnnData) -> None:
        nonlocal n_cells, n_nonzero, x_sum, x_sum_sq
        X = adata.X
        n_obs = adata.n_obs
        for start in range(0, n_obs, block_cells):
            block = X[start:min(start + block_cells, n_obs)]
            dense = block.toarray() if issparse(block) else np.asarray(block)
            dense = dense.astype(np.float64, copy=False)
            x_sum += dense.sum(axis=0)
            n_nonzero += (dense > 0).sum(axis=0)
            np.multiply(dense, dense, out=dense)  # dense² no lugar
            x_sum_sq += dense.sum(axis=0)
        n_cells += n_obs

    _accumulate(e85)
    _accumulate(e95)

    mean = x_sum / n_cells
    variance = np.maximum(x_sum_sq / n_cells - mean ** 2, 0.0)  # E[x²] − E[x]²
    pct_nonzero = n_nonzero / n_cells

    return pd.DataFrame({
        "gene_name": gene_names,
        "mean": mean,
        "variance": variance,
        "pct_nonzero": pct_nonzero,
        "n_cells": n_cells,
    })


def apply_panel_gate(n_kept: int, n_removed: int, allow_extreme: bool = False) -> None:
    """T2.01-9: falha se nenhun o todos os genes são mantidos (painel vazio o
    cheio), salvo un teste sintético que declare essa intención."""
    n_genes = n_kept + n_removed
    if n_kept == 0 or n_removed == 0:
        if allow_extreme:
            print("    ⚠ Painel extremo permitido por --allow-extreme-filter "
                  "(sólo teste sintético).")
            return
        raise RuntimeError(
            f"Painel inválido: {n_kept:,} genes mantidos / {n_removed:,} removidos "
            f"(total {n_genes:,}). Ajuste --variance-threshold; ou, se se trata "
            "de um teste sintético, pase --allow-extreme-filter."
        )


# ─── Agrupamento por correlação (opcional / opt-in) ──────────────────────────

def _dense_X(adata: ad.AnnData) -> np.ndarray:
    """Matriz densa float32 do painel (só para `--group-correlated`, que já
    opera sobre o painel mantido e amostrado; nunca sobre E8.5+E9.5 completos)."""
    X = adata.X
    return X.toarray() if issparse(X) else np.asarray(X)


def group_correlated_genes(
    adata: ad.AnnData,
    threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    chunk_size: int = 2_000,
) -> dict[str, list[str]]:
    """T2.01-3 (opt-in): agrupa genes com correlação de Pearson ≥ threshold.

    Union-Find por bloques para não explodir a memória com a matriz n_genes².
    Escribe `gene_groups.csv` em `output_dir`. Es caro e não necessário para o
    decoder; por isso só se executa com `--group-correlated`.
    """
    t0 = time.perf_counter()
    gene_names = np.asarray(adata.var_names, dtype=str)
    n_genes = gene_names.size
    n_cells = adata.n_obs
    if n_genes == 0:
        raise ValueError("group_correlated_genes: painel vazio, nada que agrupar.")

    print(f"\n── Agrupamento de genes correlacionados (threshold={threshold}) ──")
    X = _dense_X(adata)  # (células × genes), painel já filtrado/amostrado
    col_mean = X.mean(axis=0)
    col_std = X.std(axis=0)
    col_std[col_std == 0] = 1.0  # salvaguarda contra std == 0
    X_z = (X - col_mean) / col_std

    parent = list(range(n_genes))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path compression
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    n_chunks = (n_genes + chunk_size - 1) // chunk_size
    pairs_checked = 0
    for ci in range(n_chunks):
        i_start, i_end = ci * chunk_size, min((ci + 1) * chunk_size, n_genes)
        Xi = X_z[:, i_start:i_end]
        for cj in range(ci, n_chunks):
            j_start, j_end = cj * chunk_size, min((cj + 1) * chunk_size, n_genes)
            Xj = X_z[:, j_start:j_end]
            corr_block = (Xi.T @ Xj) / n_cells  # Pearson con columnas z-score
            pairs_i, pairs_j = np.where(corr_block >= threshold)
            for pi, pj in zip(pairs_i, pairs_j):
                gi = i_start + int(pi)
                gj = j_start + int(pj)
                if gi == gj or (ci == cj and gi > gj):
                    continue  # diagonal + duplicados no mesmo bloco
                if find(gi) != find(gj):
                    union(gi, gj)
                pairs_checked += 1
        print(f"    bloco {ci + 1:>4}/{n_chunks} | pares correlacionados: "
              f"{pairs_checked:,}")

    root_to_members: dict[int, list[int]] = defaultdict(list)
    for i in range(n_genes):
        root_to_members[find(i)].append(i)

    gene_means = X.mean(axis=0)
    rows = []
    for members in root_to_members.values():
        size = len(members)
        member_names = gene_names[members].tolist()
        representative = member_names[int(np.argmax(gene_means[members]))]
        rows.append({
            "group_id": len(rows),
            "representative": representative,
            "members": "|".join(member_names),
            "group_size": size,
        })
    groups_df = pd.DataFrame(rows).sort_values("group_size", ascending=False)
    groups_df.to_csv(output_dir / "gene_groups.csv", index=False)

    n_multi = int((groups_df["group_size"] >= 2).sum())
    elapsed = time.perf_counter() - t0
    print(f"  💾  gene_groups.csv → {output_dir / 'gene_groups.csv'}")
    print(f"  Resumo: {n_multi:,} grupos com ≥ 2 genes | {elapsed:.1f}s")

    return {
        row["representative"]: row["members"].split("|")
        for _, row in groups_df.iterrows()
        if row["group_size"] >= 2
    }


# ─── Fingerprint do run ───────────────────────────────────────────────────────

def build_run_fingerprint(canonical: dict) -> str:
    """SHA-256 da configuración canónica (T2.01-6). Qualquier mudança de
    threshold, seed, escala, entradas, ordem de genes ou de células altera o
    fingerprint."""
    blob = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_run_tag(
    input_scale: str,
    seed: int,
    sample_cells: int,
    variance_threshold: float,
    fingerprint: str,
) -> str:
    n_str = "all" if sample_cells <= 0 else str(sample_cells)
    return (f"t2_{input_scale}_seed{seed}_n{n_str}"
            f"_vt{variance_threshold}_{fingerprint[:8]}")


def handle_existing_run(run_dir: Path, fingerprint: str, force: bool) -> bool:
    """T2.01-5: nunca sobrescribir silenciosamente una execução incompatível.

    - Sem manifest (execução incompleta)      → continua dentro.
    - Manifest idéntico + --force             → recalcula.
    - Manifest idéntico sem --force           → nada a fazer (retorna True).
    - Manifest com fingerprint distinto       → erro alto.
    """
    if not run_dir.exists():
        return False
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"    ⚠ {run_dir} existe sin manifest (execução previa incompleta); "
              f"se continua com ela.")
        return False
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    if stored.get("run_fingerprint") == fingerprint:
        if not force:
            print(f"    ✓ {run_dir} já calculado com configuración idéntica; "
                  f"nada a fazer (--force para recalcular).")
            return True
        print("    ↻ Manifest idéntico presente; recalculo por --force.")
        return False
    raise RuntimeError(
        f"El directório {run_dir} contém um manifest de outra execução "
        "(fingerprint distinto). Não se sobrescribe silenciosamente: use outro "
        "--output-dir o outro threshold/seed/entradas."
    )


# ─── Artefatos de saída ───────────────────────────────────────────────────────

def write_artifacts(
    run_dir: Path,
    stats: pd.DataFrame,
    mask_keep: np.ndarray,
) -> dict:
    """Escribe kept_genes.txt, removed_genes.csv e gene_stats.csv (T2.01-5).

    - kept_genes.txt: ordem canónica de var_names (NUNCA ordenado por variância).
    - removed_genes.csv: gene_name, variance, mean, pct_nonzero (por variância
      ascendente; columna `gene_name` compatível com subset_genes.py).
    - gene_stats.csv: todos os genes + columna booleana `kept`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    genes = stats["gene_name"].astype(str).to_numpy()

    (run_dir / "kept_genes.txt").write_text(
        "\n".join(genes[mask_keep]) + "\n", encoding="utf-8"
    )

    removed = stats.loc[~mask_keep, ["gene_name", "variance", "mean", "pct_nonzero"]]
    removed = removed.sort_values("variance", ascending=True)
    removed.to_csv(run_dir / "removed_genes.csv", index=False)

    stats_out = stats.copy()
    stats_out["kept"] = mask_keep
    stats_out.to_csv(run_dir / "gene_stats.csv", index=False)

    return {
        name: {"path": name, "sha256": sha256_file(run_dir / name)}
        for name in ARTIFACT_PATHS
    }


def write_filtered_h5ads(
    run_dir: Path,
    e85_sampled: ad.AnnData,
    e95_sampled: ad.AnnData,
    mask_keep: np.ndarray,
) -> dict:
    """T2.01-10 (exceção opt-in via --write-filtered-h5ad).

    Escribe `e85_filtered.h5ad` / `e95_filtered.h5ad` dentro de `run_dir`,
    aplicando o mesmo `mask_keep` (painel de genes) usado em kept_genes.txt,
    sobre as MESMAS células já amostradas em `sample_cells` (portanto os
    obs_names batem exactamente com `obs_names_e85_sha256` /
    `obs_names_e95_sha256` do manifest.json deste run).

    NÃO substitui kept_genes.txt como fonte de verdade do painel: isto é só
    uma conveniência para quem quer consumir os dados já filtrados sem
    reimplementar o subset. Continua desligado por padrão (T2.01-10).
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    e85_filtered = e85_sampled[:, mask_keep].copy()
    e95_filtered = e95_sampled[:, mask_keep].copy()

    e85_filtered.write_h5ad(run_dir / "e85_filtered.h5ad")
    e95_filtered.write_h5ad(run_dir / "e95_filtered.h5ad")

    print(f"  💾  e85_filtered.h5ad → {run_dir / 'e85_filtered.h5ad'} "
          f"({e85_filtered.n_obs:,} células × {e85_filtered.n_vars:,} genes)")
    print(f"  💾  e95_filtered.h5ad → {run_dir / 'e95_filtered.h5ad'} "
          f"({e95_filtered.n_obs:,} células × {e95_filtered.n_vars:,} genes)")

    return {
        name: {"path": name, "sha256": sha256_file(run_dir / name)}
        for name in FILTERED_H5AD_PATHS
    }


def build_manifest(
    args: argparse.Namespace,
    e85_path: Path,
    e95_path: Path,
    totals: dict,
    sampled: dict,
    sample_strats: dict,
    scale_max: dict,
    var_hash: str,
    obs_hashes: dict,
    n_genes: int,
    n_kept: int,
    n_removed: int,
    fingerprint: str,
    artifacts: dict,
) -> dict:
    """T2.01-6: manifest mínimo do run (entradas, células, seed, threshold,
    escala, nº mantidos/removidos, hashes de genes e células, versão)."""
    return {
        "format_version": MANIFEST_FORMAT_VERSION,
        "tool": "src/scripts/pre_processement.py",
        "stage": "T2.01-preprocessing",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_fingerprint": fingerprint,
        "inputs": {"e85": str(e85_path), "e95": str(e95_path)},
        "scale": {
            "declared": args.input_scale,
            "observed_max_e85": float(scale_max["E8.5"]),
            "observed_max_e95": float(scale_max["E9.5"]),
        },
        "sampling": {
            "strategy_e85": sample_strats["E8.5"],
            "strategy_e95": sample_strats["E9.5"],
            "cells_per_stage_requested": args.sample_cells,
            "seed": args.seed,
            "cells": {
                "e85": {"total": totals["E8.5"], "sampled": sampled["E8.5"]},
                "e95": {"total": totals["E9.5"], "sampled": sampled["E9.5"]},
            },
        },
        "filter": {
            "variance_threshold": args.variance_threshold,
            "n_genes": n_genes,
            "n_genes_kept": n_kept,
            "n_genes_removed": n_removed,
        },
        "group_correlated": bool(getattr(args, "group_correlated", False)),
        "write_filtered_h5ad": bool(getattr(args, "write_filtered_h5ad", False)),
        "hashes": {
            "var_names_sha256": var_hash,
            "obs_names_e85_sha256": obs_hashes["E8.5"],
            "obs_names_e95_sha256": obs_hashes["E9.5"],
        },
        "artifacts": artifacts,
    }


def write_manifest(run_dir: Path, manifest: dict) -> None:
    path = run_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


# ─── Pipeline principal ───────────────────────────────────────────────────────

def run_preprocessing(args: argparse.Namespace) -> Path:
    """Executa T2.01 e retorna o directório com os artefatos."""
    t0 = time.perf_counter()
    e85_path, e95_path = Path(args.e85), Path(args.e95)
    for p, lbl in ((e85_path, "E8.5"), (e95_path, "E9.5")):
        if not p.is_file():
            raise FileNotFoundError(f"{lbl}: arquivo não encontrado → {p}")

    print("╔" + "═" * 62 + "╗")
    print("║   PRÉ-PROCESSAMENTO T2.01 – Virtual Embryo Challenge       ║")
    print("╚" + "═" * 62 + "╝")
    print(f"  input_scale         = {args.input_scale}")
    print(f"  variance_threshold  = {args.variance_threshold}")
    print(f"  sample_cells/estágio= {args.sample_cells if args.sample_cells else 'TODAS (aviso!)'}")
    print(f"  seed                = {args.seed}")
    print(f"  group_correlated    = {args.group_correlated}")
    print(f"  write_filtered_h5ad = {args.write_filtered_h5ad}")

    e85 = load_dataset(e85_path, "E8.5")
    e95 = load_dataset(e95_path, "E9.5")
    totals = {"E8.5": e85.n_obs, "E9.5": e95.n_obs}

    validate_same_gene_order(e85, e95)                      # T2.01-8
    var_hash = sha256_of_sequence(e85.var_names)

    scale_max = {
        "E8.5": check_scale_consistency(e85, args.input_scale, "E8.5"),
        "E9.5": check_scale_consistency(e95, args.input_scale, "E9.5"),
    }

    print("\n── Amostragem por estágio ─────────────────────────────────────")
    e85_s, strat_e85 = sample_cells(e85, args.sample_cells, args.seed, "E8.5")
    e95_s, strat_e95 = sample_cells(e95, args.sample_cells, args.seed, "E9.5")
    sampled = {"E8.5": e85_s.n_obs, "E9.5": e95_s.n_obs}
    sample_strats = {"E8.5": strat_e85, "E9.5": strat_e95}
    obs_hashes = {
        "E8.5": sha256_of_sequence(e85_s.obs_names),
        "E9.5": sha256_of_sequence(e95_s.obs_names),
    }
    print(f"    E8.5 → {sampled['E8.5']:,} células [{strat_e85}]")
    print(f"    E9.5 → {sampled['E9.5']:,} células [{strat_e95}]")

    print("\n── Estatísticas por gene (E8.5 + E9.5, por blocos, float64) ──")
    stats = compute_gene_stats(e85_s, e95_s)                # T2.01-1/2
    n_genes = stats.shape[0]

    variance = stats["variance"].to_numpy()
    mask_keep = variance >= args.variance_threshold
    n_kept = int(mask_keep.sum())
    n_removed = n_genes - n_kept
    apply_panel_gate(n_kept, n_removed,
                     getattr(args, "allow_extreme_filter", False))  # T2.01-9

    canonical = {
        "e85": str(e85_path.resolve()),
        "e95": str(e95_path.resolve()),
        "input_scale": args.input_scale,
        "seed": args.seed,
        "sample_cells": args.sample_cells,
        "variance_threshold": args.variance_threshold,
        "var_names_sha256": var_hash,
        "obs_names_e85_sha256": obs_hashes["E8.5"],
        "obs_names_e95_sha256": obs_hashes["E9.5"],
    }
    fingerprint = build_run_fingerprint(canonical)
    tag = build_run_tag(args.input_scale, args.seed, args.sample_cells,
                        args.variance_threshold, fingerprint)
    run_dir = Path(args.output_dir) / tag

    if handle_existing_run(run_dir, fingerprint, args.force):  # T2.01-5
        return run_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_artifacts(run_dir, stats, mask_keep)  # T2.01-5/10

    if args.group_correlated:                                # T2.01-3
        kept = ad.concat(
            [e85_s[:, mask_keep].copy(), e95_s[:, mask_keep].copy()], axis=0
        )
        group_correlated_genes(kept, threshold=args.correlation_threshold,
                               output_dir=run_dir)
        artifacts["gene_groups.csv"] = {
            "path": "gene_groups.csv",
            "sha256": sha256_file(run_dir / "gene_groups.csv"),
        }

    if args.write_filtered_h5ad:                             # T2.01-10 (opt-in)
        filtered_artifacts = write_filtered_h5ads(run_dir, e85_s, e95_s, mask_keep)
        artifacts.update(filtered_artifacts)

    manifest = build_manifest(args, e85_path.resolve(), e95_path.resolve(),
                              totals, sampled, sample_strats, scale_max,
                              var_hash, obs_hashes, n_genes, n_kept, n_removed,
                              fingerprint, artifacts)
    write_manifest(run_dir, manifest)                        # T2.01-6

    print("\n" + "═" * 64)
    print("  RELATÓRIO DO PAINEL FILTRADO (T2.01)")
    print("═" * 64)
    print(f"  Genes totais     : {n_genes:,}")
    print(f"  Threshold        : {args.variance_threshold}")
    print(f"  Genes mantidos   : {n_kept:,} ({100 * n_kept / n_genes:.2f}%)")
    print(f"  Genes removidos  : {n_removed:,} ({100 * n_removed / n_genes:.2f}%)")
    print(f"  kept + removed   : {n_kept + n_removed:,} ✓")
    print(f"  Tempo total      : {time.perf_counter() - t0:.1f}s")
    print("═" * 64)
    return run_dir


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pré-processamento T2.01 – painel filtrado reproduzible "
                    "sobre E8.5 + E9.5 (Virtual Embryo Challenge).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--e85", type=Path, required=True,
                        help="Caminho para o dataset E8.5 (.h5ad)")
    parser.add_argument("--e95", type=Path, required=True,
                        help="Caminho para o dataset E9.5 (.h5ad)")
    parser.add_argument("--sample-cells", type=int, default=DEFAULT_SAMPLE_CELLS,
                        help="Células a amostrar por estágio, estratificado por "
                             "celltype (0 = todas, com aviso de memória)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Seed da amostragem (determinista)")
    parser.add_argument("--input-scale", required=True, choices=("log1p", "counts"),
                        help="Escala declarada dos datos de entrada (obrigatório; "
                             "sem heurística silenciosa)")
    parser.add_argument("--variance-threshold", type=float,
                        default=DEFAULT_VARIANCE_THRESHOLD,
                        help="Variância mínima para manter um gene")
    parser.add_argument("--group-correlated", action="store_true",
                        help="Opt-in: agrupar genes correlacionados (caro; não "
                             "necessário para o decoder)")
    parser.add_argument("--correlation-threshold", type=float,
                        default=DEFAULT_CORRELATION_THRESHOLD,
                        help="Limiar de correlação (só com --group-correlated)")
    parser.add_argument("--write-filtered-h5ad", action="store_true",
                        help="Opt-in: escreve e85_filtered.h5ad/e95_filtered.h5ad "
                             "(painel filtrado + células amostradas) dentro do "
                             "run_dir. Desligado por padrão (T2.01-10); use quando "
                             "outro script precisar consumir os dados já prontos.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Directório base; os artefatos ficam num subdirectório "
                             "identificado pela configuração")
    parser.add_argument("--force", action="store_true",
                        help="Recalcular mesmo se o manifest é idéntico")
    parser.add_argument("--allow-extreme-filter", action="store_true",
                        help="SÓLO testes sintéticos: permite painel vazio o cheio "
                             "(T2.01-9)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    run_dir = run_preprocessing(args)
    print(f"\n✅  Pronto. Artefatos em: {run_dir.resolve()}\n")
    return run_dir


if __name__ == "__main__":
    main()