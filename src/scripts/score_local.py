"""Pontuação local da Task 1 (E10.5) do Virtual Embryo Challenge, igual ao leaderboard de validação.

Uso:
    from src.scripts.score_local import score_local
    resultado = score_local("data/E105.h5ad", "minha_previsao.h5ad")

    python -m src.scripts.score_local --e105 data/E105.h5ad --e95 data/E95.h5ad --previsao pred.h5ad

Receita do servidor (reproduz as âncoras oficiais e submissões reais com 4 casas decimais):
subamostra de 10% (sorteio sem reposição, seed 0) de E9.5 e E10.5, verdade = primeira metade
(split_half) do E10.5 subamostrado, métricas arredondadas e skill oficial do veckit.
Requer: pip install veckit anndata numpy scipy h5py

Memória: os .h5ad grandes (CSC esparso) são lidos em blocos de colunas e só as linhas sorteadas
são densificadas; o teste de Mann-Whitney roda em blocos de genes. Nunca densifica E9.5/E10.5 inteiros.
"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np

N_E105, N_E95, N_GENES = 24979, 17057, 32285
MIN_CELULAS, MAX_CELULAS = 1000, 5118
PESOS = {"de_score": 0.25, "de_direction": 0.25, "mmd_u": 0.30, "variogram": 0.20}
# (piso, teto, menor_é_melhor) das âncoras oficiais T1:val
ANCORAS = {"de_score": (0.0, 0.8464, False), "de_direction": (0.0, 0.7901, False),
           "mmd_u": (0.08359, 0.00406, True), "variogram": (0.005219, 0.000158, True)}


def _subamostra(n):
    """Subamostra de 10% usada pelo servidor."""
    return np.sort(np.random.default_rng(0).choice(n, int(round(0.1 * n)), replace=False))


def _importar_veckit():
    try:
        from common import core_metrics as cm
    except ImportError as e:
        raise ImportError("o pacote veckit não está instalado: instale com: pip install veckit") from e
    return cm


def _checar_arquivo(caminho, nome):
    p = Path(caminho)
    if p.suffix != ".h5ad":
        raise ValueError(f"{nome} deve ser um arquivo .h5ad (recebido: {p.name})")
    if not p.is_file():
        raise FileNotFoundError(f"{nome} não encontrado: {p}")
    return p


def _genes_h5(caminho):
    """Lê só shape e nomes dos genes (barato), sem carregar a matriz."""
    import h5py
    with h5py.File(caminho, "r") as f:
        var = f["var"]
        col = var.attrs["_index"]
        col = col.decode() if isinstance(col, bytes) else col
        genes = [g.decode() if isinstance(g, bytes) else str(g) for g in var[col][:]]
        X = f["X"]
        n_obs = int(X.attrs["shape"][0]) if "shape" in X.attrs else int(X.shape[0])
    return n_obs, genes


def _ler_linhas(caminho, idx):
    """Devolve as linhas `idx` (na ordem dada) como float32 denso, lendo o CSC em blocos de colunas."""
    import h5py
    with h5py.File(caminho, "r") as f:
        X = f["X"]
        enc = X.attrs.get("encoding-type", "") if hasattr(X, "attrs") else ""
        enc = enc.decode() if isinstance(enc, bytes) else enc
        if enc == "csc_matrix":
            n_obs, n_var = (int(v) for v in X.attrs["shape"])
            mapa = np.full(n_obs, -1, dtype=np.int32)
            mapa[idx] = np.arange(len(idx), dtype=np.int32)
            saida = np.zeros((len(idx), n_var), dtype=np.float32)
            indptr = X["indptr"][:]
            passo = 500
            for c0 in range(0, n_var, passo):
                c1 = min(c0 + passo, n_var)
                a, b = int(indptr[c0]), int(indptr[c1])
                linhas = mapa[X["indices"][a:b]]
                dados = X["data"][a:b]
                col = np.repeat(np.arange(c0, c1), np.diff(indptr[c0:c1 + 1]))
                ok = linhas >= 0
                saida[linhas[ok], col[ok]] = dados[ok]
            return saida
    # formato diferente (CSR/denso): leitura completa e fatiamento
    import anndata as ad
    from scipy import sparse
    X = ad.read_h5ad(caminho).X
    X = X[idx]
    out = (X.toarray() if sparse.issparse(X) else np.asarray(X)).astype(np.float32)
    del X
    gc.collect()
    return out


def _validar_previsao(caminho, genes_oficiais, verbose):
    import anndata as ad
    from scipy import sparse
    ad_p = ad.read_h5ad(caminho)
    n = ad_p.n_obs
    if n < MIN_CELULAS or n > MAX_CELULAS:
        raise ValueError(f"A previsão precisa ter entre {MIN_CELULAS} e {MAX_CELULAS} células (tem {n}); "
                         f"o limite mínimo é {MIN_CELULAS} e o máximo {MAX_CELULAS} no T1:val")
    genes = [str(g) for g in ad_p.var_names]
    if genes != genes_oficiais:
        faltam = [g for g in genes_oficiais if g not in set(genes)]
        if faltam:
            raise ValueError(f"A previsão tem {len(genes)} genes e faltam {len(faltam)} genes oficiais "
                             f"(primeiros 10: {faltam[:10]}). São exigidos exatamente {N_GENES} genes.")
        if len(genes) != N_GENES or len(set(genes)) != N_GENES:
            raise ValueError(f"A previsão deve ter exatamente {N_GENES} genes únicos (tem {len(genes)}).")
        raise ValueError("Os genes da previsão estão em ordem diferente da oficial (var_names do E9.5). "
                         "Reordene com: adata = adata[:, genes_oficiais]")
    X = ad_p.X
    del ad_p
    avisos = []
    if sparse.issparse(X):
        vals = X.data
        X = X.toarray()
    else:
        X = np.asarray(X)
        vals = X
    if X.dtype != np.float32:
        avisos.append(f"dtype {X.dtype} convertido para float32")
        X = X.astype(np.float32)
    if not np.isfinite(vals).all():
        raise ValueError("A previsão contém NaN ou infinito; todos os valores devem ser finitos.")
    if (vals < 0).any():
        raise ValueError(f"A previsão contém valores negativos (mínimo {float(vals.min()):.4g}); "
                         "a matriz deve ser log1p não negativa.")
    if float(X.max()) > 20:
        avisos.append(f"valor máximo {float(X.max()):.1f} > 20: a escala parece não ser log1p")
    amostra = np.random.default_rng(0).choice(n, min(n, 200), replace=False)
    soma = np.expm1(X[amostra].astype(np.float64)).sum(1)
    mediana = float(np.median(soma))
    if not 5000 <= mediana <= 20000:
        avisos.append(f"soma expm1 por célula (mediana {mediana:.0f}) longe de 10.000: normalize com "
                      "sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)")
    if verbose:
        for a in avisos:
            print(f"AVISO: {a}")
    return X, avisos


def _mwu_em_blocos(original, bloco=1000):
    """Mann-Whitney coluna a coluna em blocos de genes (p-valores idênticos, memória bem menor)."""
    def f(A, B, axis=0, alternative="two-sided", **kw):
        if axis != 0:
            return original(A, B, axis=axis, alternative=alternative, **kw)
        partes = [original(A[:, s:s + bloco], B[:, s:s + bloco], axis=0, alternative=alternative, **kw)
                  for s in range(0, A.shape[1], bloco)]
        return np.concatenate([p[0] for p in partes]), np.concatenate([p[1] for p in partes])
    return f


def _pico_rss_mb():
    try:
        import resource
        import sys
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024
    except Exception:  # Windows
        return None


def score_local(local_e105, previsao, local_e95="data/E95.h5ad", verbose=True) -> dict:
    """Calcula o placar T1:val (E10.5) localmente. Retorna total, componentes (0-100), métricas brutas e info."""
    # 1) validações baratas primeiro
    cm = _importar_veckit()
    p_e105 = _checar_arquivo(local_e105, "local_e105")
    p_e95 = _checar_arquivo(local_e95, "local_e95")
    p_prev = _checar_arquivo(previsao, "previsao")
    n105, genes105 = _genes_h5(p_e105)
    n95, genes95 = _genes_h5(p_e95)
    if n105 != N_E105 or n95 != N_E95 or len(genes95) != N_GENES or len(genes105) != N_GENES:
        raise ValueError(f"Arquivos inesperados: E10.5 tem {n105} células (esperado {N_E105}) e E9.5 tem {n95} "
                         f"(esperado {N_E95}), com {len(genes95)} genes (esperado {N_GENES}). O servidor assume "
                         "os arquivos oficiais liberados pela competição.")
    pred, avisos = _validar_previsao(p_prev, genes95, verbose)

    # 2) referência e verdade, lendo só as linhas sorteadas
    ref = _ler_linhas(p_e95, _subamostra(N_E95))
    sub = _subamostra(N_E105)
    a, _ = cm.split_half(len(sub), 0)
    verdade = _ler_linhas(p_e105, sub[a])  # mantém a ordem de split_half (afeta o MMD)
    gc.collect()

    # 3) métricas oficiais (Mann-Whitney em blocos, restaurado ao final)
    import scipy.stats
    original = scipy.stats.mannwhitneyu
    scipy.stats.mannwhitneyu = _mwu_em_blocos(original)
    try:
        de = cm.de_score(pred, verdade, ref)
        ds = de["score"]
        ds = round(ds, 4) if ds is not None and np.isfinite(ds) else None
        brutas = {"de_score": ds,
                  "de_direction": round(cm.de_direction(pred, verdade, ref), 4),
                  "mmd_u": round(cm.mmd_unbiased(pred, verdade, seed=0), 5),
                  "variogram": round(cm.variogram_score(pred, verdade, seed=0), 6)}
    finally:
        scipy.stats.mannwhitneyu = original

    comp = {}
    for k, (piso, teto, menor) in ANCORAS.items():
        if brutas[k] is None:
            print("AVISO: de_score indisponível (NaN); skill tratado como 0")
            comp[k] = 0.0
        else:
            comp[k] = 100 * cm.skill(brutas[k], piso, teto, lower_is_better=menor)
    total = float(sum(PESOS[k] * comp[k] for k in PESOS))
    info = {"celulas_previsao": int(pred.shape[0]), "celulas_referencia": int(ref.shape[0]),
            "celulas_verdade": int(verdade.shape[0]), "avisos": avisos, "pico_rss_mb": _pico_rss_mb()}
    res = {"total": total, "componentes": comp, "metricas_brutas": brutas, "info": info}
    if verbose:
        print(f"\n{'Métrica':<14}{'Peso':>6}{'Skill (0-100)':>16}{'Valor bruto':>14}")
        for k in PESOS:
            print(f"{k:<14}{PESOS[k]:>6.2f}{comp[k]:>16.4f}{str(brutas[k]):>14}")
        print(f"{'TOTAL':<14}{'':>6}{total:>16.4f}")
        print("Nota: copy_last (copiar o E9.5) = 50 nesta escala.")
        if info["pico_rss_mb"]:
            print(f"Pico de memória (RSS): {info['pico_rss_mb']:.0f} MB")
    return res


def main():
    ap = argparse.ArgumentParser(description="Placar local T1:val (E10.5) do Virtual Embryo Challenge")
    ap.add_argument("--e105", default="data/E105.h5ad")
    ap.add_argument("--e95", default="data/E95.h5ad")
    ap.add_argument("--previsao", required=True)
    a = ap.parse_args()
    score_local(a.e105, a.previsao, a.e95)


if __name__ == "__main__":
    main()
