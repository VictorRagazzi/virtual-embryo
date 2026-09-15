"""Pareamento probabilístico de células E8.5 -> E9.5 via Optimal Transport
não-balanceado, usando moscot (moscot.problems.time.TemporalProblem).

IMPORTANTE: o custo/pareamento aqui NÃO é restrito por obs['celltype']. O
problema é preparado usando apenas o embedding conjunto (`joint_attr`) como
espaço de custo - nenhuma marginal, custo ou filtro depende do celltype. A
transição entre tipos celulares ao longo do desenvolvimento é, portanto, algo
que o transporte pode livremente capturar (e não algo eliminado a priori).

Uso (via uv, a partir da raiz do projeto):
#     uv run python -m src.approaches.llm.T1.moscot_apply.ot_pairing --e85 data/preprocessing/t2_log1p_seed0_n16500_vt0.05_d65fe010/e85_filtered.h5ad --e95 data/preprocessing/t2_log1p_seed0_n16500_vt0.05_d65fe010/e95_filtered.h5ad --out-dir data --epsilon 0.01 --tau-a 0.95 --tau-b 0.95 --pairing-mode sample --n-samples 1

    uv run python -m src.approaches.llm.T1.moscot_apply.ot_pairing \\
        --e85 data/E85.h5ad \\
        --e95 data/E95.h5ad \\
        --out-dir data \\
        --epsilon 0.01 --tau-a 0.95 --tau-b 0.95 \\
        --pairing-mode sample --n-samples 1

Saídas em `--out-dir` (default: data/):
    e85_e95_ot_pairs.h5ad   - AnnData pareado (X=E8.5, layers['target']=E9.5)
    e85_e95_ot_pairs.csv    - tabela de pares (source_cell, target_cell, weight, ...)
    e85_e95_transport.npz   - matriz de transporte esparsificada (top-k por linha)
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from moscot.problems.time import TemporalProblem

from .build_dataset import build_paired_anndata, save_outputs
from .data_utils import prepare_adata
from .pairing_sampler import transport_to_pairs

logger = logging.getLogger(__name__)


def solve_ot(
    adata: ad.AnnData,
    time_e85: float = 8.5,
    time_e95: float = 9.5,
    joint_attr: str = "X_pca_joint",
    epsilon: float = 0.01,
    tau_a: float = 0.95,
    tau_b: float = 0.95,
    rank: int = -1,
    scale_cost: str = "mean",
) -> TemporalProblem:
    """Constrói e resolve o TemporalProblem do moscot entre E8.5 (source) e E9.5 (target).

    - tau_a / tau_b < 1  => OT NÃO-BALANCEADO (a massa de cada célula não precisa
      ser conservada exatamente; permite que células "nasçam"/"desapareçam" no
      acoplamento, o que é apropriado aqui já que não há rastreamento real de
      linhagem entre os timepoints e os tamanhos de população/tipo mudam).
      tau_a = tau_b = 1.0 recairia no OT balanceado clássico.
    - Nenhum argumento relacionado a celltype é passado: `a`/`b` (priors de
      marginal) ficam em None -> uniformes; o custo usa apenas `joint_attr`.
    - `rank=-1` usa Sinkhorn de rank completo; para datasets grandes (~16k x 16k),
      considere `rank` positivo (OT de baixo posto) para acelerar/reduzir memória.
    """
    tp = TemporalProblem(adata)
    tp = tp.prepare(
        time_key="time",
        joint_attr=joint_attr,
        policy="sequential",
        cost="sq_euclidean",
    )
    logger.info(
        "Resolvendo OT não-balanceado (epsilon=%.4g, tau_a=%.3f, tau_b=%.3f, rank=%d)...",
        epsilon, tau_a, tau_b, rank,
    )
    tp = tp.solve(
        epsilon=epsilon,
        tau_a=tau_a,
        tau_b=tau_b,
        rank=rank,
        scale_cost=scale_cost,
    )
    key = (time_e85, time_e95)
    if key not in tp.solutions:
        raise RuntimeError(f"Chave {key} não encontrada em tp.solutions: {list(tp.solutions.keys())}")
    if not tp.solutions[key].converged:
        logger.warning("O solver do OT não reportou convergência para %s - revise epsilon/iterações.", key)
    return tp


def get_transport_matrix(tp: TemporalProblem, time_e85: float, time_e95: float) -> np.ndarray:
    """Extrai a matriz de transporte densa (n_source x n_target) do problema resolvido."""
    solution = tp.solutions[(time_e85, time_e95)]
    T = np.asarray(solution.transport_matrix)
    return T


def sparsify_transport(T: np.ndarray, top_k: int = 50) -> sparse.csr_matrix:
    """Mantém apenas os `top_k` maiores pesos por linha (para salvar em disco de
    forma compacta - uma matriz densa 16k x 16k em float32 já ocupa ~1GB)."""
    n_src, n_tgt = T.shape
    top_k = min(top_k, n_tgt)
    rows, cols, vals = [], [], []
    part_idx = np.argpartition(-T, top_k - 1, axis=1)[:, :top_k]
    for i in range(n_src):
        js = part_idx[i]
        ws = T[i, js]
        mask = ws > 0
        rows.extend([i] * mask.sum())
        cols.extend(js[mask].tolist())
        vals.extend(ws[mask].tolist())
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n_src, n_tgt))


def run_pipeline(
    e85_path: str,
    e95_path: str,
    out_dir: str = "data",
    time_e85: float = 8.5,
    time_e95: float = 9.5,
    skip_normalization: bool = False,
    n_pcs: int = 50,
    use_hvg: bool = True,
    n_top_genes: int = 2000,
    epsilon: float = 0.01,
    tau_a: float = 0.95,
    tau_b: float = 0.95,
    rank: int = -1,
    scale_cost: str = "mean",
    pairing_mode: str = "sample",
    n_samples: int = 1,
    min_row_mass: float = 1e-8,
    random_state: int = 0,
    transport_top_k: int = 50,
    prefix: str = "e85_e95_ot_pairs",
) -> dict:
    """Executa o pipeline completo: carregar -> PCA conjunta -> OT -> pareamento -> salvar."""

    adata = prepare_adata(
        e85_path=e85_path,
        e95_path=e95_path,
        skip_normalization=skip_normalization,
        recompute_pca=True,
        n_pcs=n_pcs,
        use_hvg=use_hvg,
        n_top_genes=n_top_genes,
    )

    tp = solve_ot(
        adata,
        time_e85=time_e85,
        time_e95=time_e95,
        joint_attr="X_pca_joint",
        epsilon=epsilon,
        tau_a=tau_a,
        tau_b=tau_b,
        rank=rank,
        scale_cost=scale_cost,
    )

    T = get_transport_matrix(tp, time_e85, time_e95)

    mask_src = (adata.obs["time"] == time_e85).values
    mask_tgt = (adata.obs["time"] == time_e95).values
    source_names = adata.obs_names[mask_src]
    target_names = adata.obs_names[mask_tgt]
    assert T.shape == (len(source_names), len(target_names)), (
        f"Shape da matriz de transporte {T.shape} não bate com "
        f"({len(source_names)}, {len(target_names)})."
    )

    pairs = transport_to_pairs(
        T,
        source_names=source_names,
        target_names=target_names,
        mode=pairing_mode,
        n_samples=n_samples,
        min_row_mass=min_row_mass,
        random_state=random_state,
    )

    paired = build_paired_anndata(adata, pairs)
    out_paths = save_outputs(paired, pairs, out_dir=out_dir, prefix=prefix)

    T_sparse = sparsify_transport(T, top_k=transport_top_k)
    transport_path = Path(out_dir) / f"{prefix}_transport.npz"
    sparse.save_npz(transport_path, T_sparse)
    out_paths["transport_npz"] = str(transport_path)
    logger.info("Matriz de transporte esparsificada (top-%d/linha) salva em: %s", transport_top_k, transport_path)

    return out_paths


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pareamento probabilístico E8.5->E9.5 via Optimal Transport não-balanceado (moscot)."
    )
    p.add_argument("--e85", type=str, default="data/E85.h5ad", help="Caminho do .h5ad de E8.5")
    p.add_argument("--e95", type=str, default="data/E95.h5ad", help="Caminho do .h5ad de E9.5")
    p.add_argument("--out-dir", type=str, default="data", help="Diretório de saída")
    p.add_argument("--prefix", type=str, default="e85_e95_ot_pairs", help="Prefixo dos arquivos de saída")

    p.add_argument("--time-e85", type=float, default=8.5)
    p.add_argument("--time-e95", type=float, default=9.5)

    p.add_argument("--skip-normalization", action="store_true",
                    help="Usar se os dados de entrada já estiverem normalizados/log1p.")
    p.add_argument("--n-pcs", type=int, default=50)
    p.add_argument("--no-hvg", dest="use_hvg", action="store_false",
                    help="Não restringir a PCA conjunta a genes altamente variáveis.")
    p.add_argument("--n-top-genes", type=int, default=2000)

    p.add_argument("--epsilon", type=float, default=0.01, help="Regularização entrópica do Sinkhorn.")
    p.add_argument("--tau-a", type=float, default=0.95, help="Relaxamento da marginal fonte (1.0 = balanceado).")
    p.add_argument("--tau-b", type=float, default=0.95, help="Relaxamento da marginal alvo (1.0 = balanceado).")
    p.add_argument("--rank", type=int, default=-1, help="-1 = full-rank; >0 = OT de baixo posto (mais rápido/leve).")
    p.add_argument("--scale-cost", type=str, default="mean")

    p.add_argument("--pairing-mode", type=str, default="sample", choices=["sample", "argmax", "top_k"])
    p.add_argument("--n-samples", type=int, default=1,
                    help="Nº de alvos por célula fonte (1 = um par por célula, como pedido).")
    p.add_argument("--min-row-mass", type=float, default=1e-8)
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--transport-top-k", type=int, default=50,
                    help="Nº de maiores pesos por linha mantidos ao salvar a matriz de transporte esparsa.")

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if not args.verbose else logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out_paths = run_pipeline(
        e85_path=args.e85,
        e95_path=args.e95,
        out_dir=args.out_dir,
        time_e85=args.time_e85,
        time_e95=args.time_e95,
        skip_normalization=args.skip_normalization,
        n_pcs=args.n_pcs,
        use_hvg=args.use_hvg,
        n_top_genes=args.n_top_genes,
        epsilon=args.epsilon,
        tau_a=args.tau_a,
        tau_b=args.tau_b,
        rank=args.rank,
        scale_cost=args.scale_cost,
        pairing_mode=args.pairing_mode,
        n_samples=args.n_samples,
        min_row_mass=args.min_row_mass,
        random_state=args.random_state,
        transport_top_k=args.transport_top_k,
        prefix=args.prefix,
    )
    print("\nArquivos gerados:")
    for k, v in out_paths.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()