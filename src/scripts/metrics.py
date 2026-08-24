"""
metrics.py — Métricas oficiais da Task 1 (Temporal) do Virtual Embryo Challenge (NeurIPS 2026).

Fonte canônica
--------------
Repositório oficial:   https://github.com/aristoteleo/veckit
Arquivo de referência: common/core_metrics.py  (fornecido pela organização)
Página da tarefa:      https://virtualembryo.ai/challenge/tasks/temporal

As quatro métricas da Task 1 e seus pesos no leaderboard
---------------------------------------------------------
| Sigla | Nome oficial              | Peso | Direção        |
|-------|---------------------------|------|----------------|
| DES   | de_score                  | 25 % | higher is better |
| DCS   | de_direction              | 25 % | higher is better |
| MMD   | mmd_unbiased              | 30 % | lower is better  |
| CSS   | variogram_score           | 20 % | lower is better  |

Convenção de dados
------------------
Todas as matrizes de expressão gênica devem estar em escala
log1p-normalizada: log(1 + 1e4 × counts / library_size).
Isso corresponde ao formato dos arquivos .h5ad liberados pela competição
(target_sum = 1e4, scanpy/AnnData padrão).

Âncoras de calibração para skill() — Task 1, split de validação (E10.5)
-------------------------------------------------------------------------
Estas âncoras são estimadas a partir da documentação oficial. Confirme
os valores exatos na página de avaliação antes de usá-las para comparar
com o leaderboard:
  floor   = baseline "copy_last" (submeter o estágio anterior como previsão)
  ceiling = split-half oracle (uma metade do target avaliada contra a outra)

  de_score     → floor ≈ 0.00,    ceiling ≈ 0.846   (higher is better)
  de_direction → floor ≈ 0.00,    ceiling ≈ 0.790   (higher is better)
  mmd_u        → floor ≈ 0.0836,  ceiling ≈ 0.00406 (lower is better)
  variogram    → floor ≈ 0.00522, ceiling ≈ 0.000158 (lower is better)

Atenção: essas âncoras não estão embutidas no evaluate_all() porque podem
mudar entre versões do leaderboard. Forneça-as manualmente em skill().
"""

from __future__ import annotations

from typing import Union

import numpy as np
from scipy import sparse

# ---------------------------------------------------------------------------
# Tipo aceito pelas funções públicas
# ---------------------------------------------------------------------------
Matrix = Union[np.ndarray, "sparse.spmatrix"]

# ---------------------------------------------------------------------------
# Constantes globais — idênticas ao core_metrics.py oficial
# ---------------------------------------------------------------------------

#: Deslocamento mínimo em log-expressão média para um gene ser considerado DE.
MIN_LFC: float = 0.25

#: Piso de severity_slope para previsões sem resposta (log(1e-3) ≈ -6.9079).
SEVERITY_LOG_WORST: float = float(np.log(1e-3))


# ===========================================================================
# Utilitários internos
# ===========================================================================

def _to_dense(X: Matrix) -> np.ndarray:
    """Converte matriz esparsa para densa; arrays numpy passam sem cópia."""
    return X.toarray() if sparse.issparse(X) else np.asarray(X)


def _pseudobulk(X: Matrix) -> np.ndarray:
    """Média de células (eixo 0): shape (n_cells, n_genes) → (n_genes,)."""
    return np.asarray(_to_dense(X).mean(0)).ravel()


def _ranks(v: np.ndarray) -> np.ndarray:
    """Rank-transform com tratamento de empates (scipy.stats.rankdata)."""
    from scipy.stats import rankdata
    return rankdata(np.asarray(v, float))


def _pairwise_euclidean(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distância Euclidiana par-a-par via identidade BLAS: ||a-b||² = ||a||² + ||b||² - 2 a·b.

    Matematicamente idêntico a scipy.spatial.distance.cdist, mas ~95× mais
    rápido em matrizes largas (ex.: 1500×1500×32285) porque roteia pelo
    SGEMM otimizado do BLAS ao invés do loop C serial do cdist.
    """
    aa = np.einsum("ij,ij->i", A, A)
    bb = np.einsum("ij,ij->i", B, B)
    d2 = aa[:, None] + bb[None, :] - 2.0 * (A @ B.T)
    np.maximum(d2, 0, out=d2)   # cancelamento fp pode dar valores levemente negativos
    return np.sqrt(d2, out=d2)


# ===========================================================================
# Funções auxiliares das métricas
# ===========================================================================

def de_genes(
    X_cond: Matrix,
    X_ref: Matrix,
    alpha: float = 0.05,
    min_lfc: float = MIN_LFC,
    min_cells: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Genes DE significativos, separados por direção.

    Usa Wilcoxon rank-sum (Mann-Whitney U, bilateral) + correção FDR
    Benjamini-Hochberg em ``alpha`` E um piso de efeito ``|lfc| >= min_lfc``
    na log-expressão média.

    Nota do código oficial: o passo FDR é, na prática, inerte após aplicar
    o piso de efeito nos dados desta competição (verificado 107/107 genes em
    T1 e 78/78 em T3). O conjunto retornado é equivalente a {|lfc| >= min_lfc},
    mas o passo estatístico permanece porque pode ser relevante com contagens
    de células menores ou painéis mais amplos.

    Parameters
    ----------
    X_cond:
        Matriz de expressão da condição de interesse (n_cells × n_genes).
    X_ref:
        Matriz de expressão da condição de referência (n_cells × n_genes).
    alpha:
        Nível de significância para o FDR BH.
    min_lfc:
        Piso mínimo de |log-fold-change| na pseudobulk.
    min_cells:
        Número mínimo de células em cada grupo; retorna conjuntos vazios
        se não atingido.

    Returns
    -------
    up_indices : np.ndarray[int]
        Índices de colunas (genes) com lfc > 0 e significativos.
    down_indices : np.ndarray[int]
        Índices de colunas (genes) com lfc < 0 e significativos.
    lfc : np.ndarray[float]
        Log-fold-change completo (len = n_genes), positivo = up em X_cond.
    """
    from scipy.stats import mannwhitneyu

    A, B = _to_dense(X_cond), _to_dense(X_ref)
    lfc = _pseudobulk(A) - _pseudobulk(B)

    if A.shape[0] < min_cells or B.shape[0] < min_cells:
        return np.array([], int), np.array([], int), lfc

    with np.errstate(all="ignore"):
        _stat, p = mannwhitneyu(A, B, axis=0, alternative="two-sided")  # correção de empates por padrão

    p = np.nan_to_num(p, nan=1.0)
    order = np.argsort(p)
    m = len(p)
    below = p[order] <= alpha * (np.arange(1, m + 1) / m)   # Benjamini-Hochberg
    k = int(np.max(np.where(below)[0]) + 1) if below.any() else 0
    sig = order[:k]

    if min_lfc > 0:
        sig = sig[np.abs(lfc[sig]) >= min_lfc]

    return sig[lfc[sig] > 0], sig[lfc[sig] < 0], lfc


def _signed_overlap(
    lfc_pred: np.ndarray,
    up_true: np.ndarray,
    dn_true: np.ndarray,
) -> tuple[float, int]:
    """Sobreposição baseada em ranking, separada por sinal.

    Pega os top-|up_true| genes da previsão por logFC positivo e os
    bottom-|dn_true| por logFC negativo, e intersecta up com up e down
    com down.

    Propriedades-chave (que fecham os ataques descritos no cabeçalho do módulo):
    - **Rank-based**: multiplicar o delta da previsão por qualquer constante
      positiva não altera o resultado.
    - **Sign-split**: negar a previsão move seus genes-up para o slot de down,
      onde eles não coincidem com os up do ground truth.

    Returns
    -------
    overlap_fraction : float
        Hits / (n_up + n_dn). ``nan`` se o conjunto verdadeiro for vazio.
    n_true : int
        n_up + n_dn (zero quando o conjunto é vazio).
    """
    n_up, n_dn = len(up_true), len(dn_true)
    if n_up + n_dn == 0:
        return float("nan"), 0

    order = np.argsort(-np.asarray(lfc_pred, float))         # mais positivo primeiro
    pred_up = order[:n_up]
    pred_dn = order[len(order) - n_dn:] if n_dn else np.array([], int)

    hits = len(np.intersect1d(pred_up, up_true)) + len(np.intersect1d(pred_dn, dn_true))
    return hits / (n_up + n_dn), n_up + n_dn


# ===========================================================================
# Métrica 1 — DES: Differential Expression Score  (peso 25%)
# ===========================================================================

def de_score(
    pred_X: Matrix,
    true_X: Matrix,
    ref_X: Matrix,
    alpha: float = 0.05,
    min_lfc: float = MIN_LFC,
) -> dict:
    """**DES — Differential Expression Score** (peso 25 %, higher is better).

    Mede se a previsão nomeia os genes que realmente mudam, e os move na
    direção correta, **além do que uma submissão sem informação biológica
    conseguiria**.

    Algoritmo (idêntico ao oficial):
    1. O ground truth define quantos genes sobem (``n_up``) e quantos descem
       (``n_dn``) em relação à referência, via Wilcoxon + BH + piso de efeito.
    2. A previsão é avaliada pedindo exatamente ``n_up + n_dn`` genes,
       ranqueados pelo seu próprio logFC; up é comparado com up e down com down.
    3. O null **empírico** é medido como a melhor sobreposição obtida
       ranqueando genes pelo nível de expressão médio da referência — exatamente
       o que o ataque "resubmeter o estágio de entrada escalado" produz — em
       **ambas** as orientações (+pb_ref e -pb_ref), tomando o máximo.
    4. Score final:
       ``de_score = (overlap - chance) / (1 - chance)``
       0 = igual ao null empírico; negativo = pior; 1 = conjuntos DE perfeitos.

    Parameters
    ----------
    pred_X:
        Expressão prevista (n_cells_pred × n_genes). log1p-normalizada.
    true_X:
        Expressão observada do target (n_cells_true × n_genes). log1p-normalizada.
    ref_X:
        Expressão da condição de referência/entrada (n_cells_ref × n_genes).
        log1p-normalizada.
    alpha:
        Nível FDR para a seleção de genes DE.
    min_lfc:
        Piso de efeito mínimo em log-expressão.

    Returns
    -------
    dict com as chaves:
        ``score``          — DES corrigido pelo null empírico (métrica principal).
        ``raw``            — sobreposição bruta sem correção.
        ``chance``         — null empírico (nível de expressão da referência).
        ``chance_uniform`` — null hipergeométrico n_true/G (reportado por
                             compatibilidade com versões anteriores).
        ``n_up``           — genes up no ground truth.
        ``n_dn``           — genes down no ground truth.
        ``n_true``         — total de genes DE no ground truth.

    Notes
    -----
    Pré-processamento necessário
        Os dados **devem** estar em log1p-normalizada (target_sum=1e4).
        Não aplique normalização adicional antes de chamar esta função.

    Tratamento de ausência de mudança
        Se a previsão não tiver mudança detectável (std(dp) < 1e-2 × std(dt)),
        retorna score=0.0 diretamente, evitando divisão por zero e ruído float32.
    """
    up_t, dn_t, _ = de_genes(true_X, ref_X, alpha, min_lfc)
    G = int(true_X.shape[1])
    nan = float("nan")
    n_true = len(up_t) + len(dn_t)

    out: dict = {
        "score": nan, "raw": nan, "chance": nan, "chance_uniform": nan,
        "n_up": len(up_t), "n_dn": len(dn_t), "n_true": n_true,
    }
    if n_true == 0:
        return out

    dp = _pseudobulk(pred_X) - _pseudobulk(ref_X)
    dt = _pseudobulk(true_X) - _pseudobulk(ref_X)

    # Null empírico em AMBAS as orientações; toma o melhor (mais favorável ao atacante)
    pb_ref = _pseudobulk(ref_X)
    c_up, _ = _signed_overlap(pb_ref, up_t, dn_t)
    c_dn, _ = _signed_overlap(-pb_ref, up_t, dn_t)
    chance = max(c_up, c_dn)
    out["chance"] = float(chance)
    out["chance_uniform"] = float(n_true / G)

    if np.std(dt) < 1e-12:
        return out                                 # ground truth sem resposta: board indefinido

    if np.std(dp) < 1e-2 * np.std(dt):            # previsão sem mudança detectável
        out.update(score=0.0, raw=0.0)
        return out

    raw, _ = _signed_overlap(dp, up_t, dn_t)
    out["raw"] = float(raw)
    out["score"] = float((raw - chance) / (1 - chance)) if chance < 1 else nan
    return out


# ===========================================================================
# Métrica 2 — DCS: Directional Concordance Score  (peso 25%)
# ===========================================================================

def de_direction(
    pred_X: Matrix,
    true_X: Matrix,
    ref_X: Matrix,
) -> float:
    """**DCS — Directional Concordance Score** (peso 25 %, higher is better).

    Correlação parcial de ranks entre o logFC previsto e o logFC observado,
    **controlando pelo nível de expressão da referência**.

    Algoritmo (idêntico ao oficial):
    1. Computa pseudo-bulk delta: dp = pb(pred) - pb(ref), dt = pb(true) - pb(ref).
    2. Rank-transforma dp, dt e pb(ref).
    3. Calcula a correlação parcial de dp com dt, partialling out pb(ref):
       ``r = (C[0,1] - C[0,2]*C[1,2]) / sqrt((1 - C[0,2]²)(1 - C[1,2]²))``
       onde C é a matriz de correlação entre [rank_dp, rank_dt, rank_pb_ref].

    Por que o controle é necessário
        Sem ele, submeter 0.99 × referência pontuava 0.31 (T1) e 0.54 (T3),
        superando todos os modelos reais. O delta desse ataque é exatamente
        proporcional a -pb(ref), e a mudança biológica real também é correlacionada
        com o nível de expressão. O controle remove esse componente compartilhado.

    Parameters
    ----------
    pred_X:
        Expressão prevista (n_cells_pred × n_genes).
    true_X:
        Expressão observada do target (n_cells_true × n_genes).
    ref_X:
        Expressão da referência (n_cells_ref × n_genes).

    Returns
    -------
    float
        Correlação parcial de ranks em [-1, 1].
        0.0 se a previsão não tiver mudança ou for (anti)colinear com pb(ref).

    Notes
    -----
    Casos especiais
        - Sem mudança na previsão (std(dp) < 1e-2 × std(dt)): retorna 0.0.
        - Previsão colinear com pb(ref) (|corr(rank_dp, rank_pb_ref)| ≈ 1):
          retorna 0.0 em vez de ruído float32 amplificado pela divisão 0/0.
    """
    dp = _pseudobulk(pred_X) - _pseudobulk(ref_X)
    dt = _pseudobulk(true_X) - _pseudobulk(ref_X)

    if np.std(dt) < 1e-12 or np.std(dp) < 1e-2 * np.std(dt):
        return 0.0

    rp, rt, rr = _ranks(dp), _ranks(dt), _ranks(_pseudobulk(ref_X))
    C = np.corrcoef(np.vstack([rp, rt, rr]))

    if not np.all(np.isfinite(C)):
        return 0.0

    # Previsão (anti)colinear com nível de expressão da referência → sem informação adicional
    if 1.0 - abs(C[0, 2]) < 1e-6:
        return 0.0

    num = C[0, 1] - C[0, 2] * C[1, 2]
    den = np.sqrt(max((1 - C[0, 2] ** 2) * (1 - C[1, 2] ** 2), 1e-12))
    r = num / den
    return float(np.clip(r, -1.0, 1.0)) if np.isfinite(r) else 0.0


# ===========================================================================
# Métrica 3 — MMD: Maximum Mean Discrepancy  (peso 30%)
# ===========================================================================

def mmd_unbiased(
    pred_X: Matrix,
    true_X: Matrix,
    n: int = 2000,
    n_pc: int = 30,
    seed: int = 0,
    scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
) -> float:
    """**MMD — Maximum Mean Discrepancy** (peso 30 %, lower is better).

    MMD² não-viesado com múltiplos kernels RBF, em espaço PCA ajustado
    **apenas** nas células observadas do target.

    Algoritmo (idêntico ao oficial):
    1. Ajusta PCA com ``n_pc`` componentes nas células do ground truth.
    2. Sub-amostra até ``n`` células de cada lado (aleatório, sem reposição).
    3. Projeta ambos no espaço PCA.
    4. Estima a largura de banda γ₀ pelo heurístico da mediana:
       ``γ₀ = 1 / median(||b_i - b_j||²  para i ≠ j)``
    5. Para cada escala ``s ∈ scales``, usa γ = γ₀ × s e computa MMD²
       com estimador **não-viesado** (diagonal zerada, denominadores n(n-1)):
       ``Kaa_sum / [na(na-1)] + Kbb_sum / [nb(nb-1)] - 2 mean(Kab)``
    6. Retorna a média sobre as escalas.

    Por que multi-kernel
        Um único kernel cria um ponto cego: uma submissão pode se posicionar
        no ângulo morto de uma largura de banda específica. A mistura de 5
        escalas fecha esse ângulo.

    Por que não-viesado
        O estimador viesado mantém a diagonal (Kii = 1 para RBF), que somou
        ~2/n ao valor do oracle e fez o teto do skill ser 0.0004 em vez de 0,
        distorcendo toda a normalização.

    Limitação residual (documentada no código oficial)
        PCA para ``n_pc`` componentes é uma projeção linear; qualquer estrutura
        no subespaço descartado é invisível. Use ``energy_distance()`` para
        uma checagem no espaço completo.

    Parameters
    ----------
    pred_X:
        Expressão prevista (n_cells_pred × n_genes).
    true_X:
        Expressão observada do target (n_cells_true × n_genes).
    n:
        Tamanho máximo da sub-amostra por lado.
    n_pc:
        Número de componentes principais (espaço de avaliação).
    seed:
        Semente do RNG para reprodutibilidade.
    scales:
        Fatores multiplicativos da largura de banda γ₀.

    Returns
    -------
    float
        MMD² médio sobre as escalas. 0 = distribuições idênticas; maior = mais distante.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import rbf_kernel

    pred_X, true_X = _to_dense(pred_X), _to_dense(true_X)
    rng = np.random.default_rng(seed)

    # PCA ajustado SOMENTE no ground truth
    pca = PCA(n_components=min(n_pc, true_X.shape[1]), random_state=0).fit(true_X)

    A = pca.transform(pred_X[rng.choice(pred_X.shape[0], min(n, pred_X.shape[0]), replace=False)])
    B = pca.transform(true_X[rng.choice(true_X.shape[0], min(n, true_X.shape[0]), replace=False)])

    # Heurístico da mediana para largura de banda base (calculado apenas em B)
    d2 = np.sum((B[:, None] - B[None, :]) ** 2, axis=-1)
    gamma0 = 1.0 / (np.median(d2[d2 > 0]) + 1e-9)

    na, nb = A.shape[0], B.shape[0]
    total = 0.0
    for s in scales:
        g = gamma0 * s
        Kaa = rbf_kernel(A, A, g)
        Kbb = rbf_kernel(B, B, g)
        Kab = rbf_kernel(A, B, g)
        np.fill_diagonal(Kaa, 0.0)    # estimador NÃO-viesado: remove diagonal
        np.fill_diagonal(Kbb, 0.0)
        total += (
            Kaa.sum() / (na * (na - 1))
            + Kbb.sum() / (nb * (nb - 1))
            - 2 * Kab.mean()
        )
    return float(total / len(scales))


# ===========================================================================
# Métrica 4 — CSS: Co-expression Structure Score  (peso 20%)
# ===========================================================================

def variogram_score(
    pred_X: Matrix,
    true_X: Matrix,
    n_pairs: int = 20_000,
    n_cells: int = 1_500,
    p: float = 0.5,
    seed: int = 0,
) -> float:
    """**CSS — Co-expression Structure Score / Variogram Score** (peso 20 %, lower is better).

    Única métrica da suíte que restringe a estrutura conjunta gene-gene.

    Algoritmo (idêntico ao oficial):
    1. Sub-amostra até ``n_cells`` células de cada lado.
    2. Amostra aleatoriamente ``n_pairs`` pares de genes (i, j) com i ≠ j.
    3. Para cada par, calcula o variograma empírico fracionário (p=0.5):
       ``v(i,j) = E[ |x_i - x_j|^p ]``
    4. Retorna o MSE entre as duas matrizes de variograma:
       ``VS = mean_over_pairs ( v_pred(i,j) - v_true(i,j) )²``

    Por que este termo é necessário
        Todas as outras métricas da suíte operam em marginais por gene,
        médias 500-dim, ou distâncias de kernel em espaço de 30 PCs.
        Uma submissão com marginais perfeitas e co-expressão completamente
        embaralhada passa em todas elas. O variogram score é o componente
        da literatura de proper scoring rules que detecta exatamente esse
        tipo de falha.

    Parameters
    ----------
    pred_X:
        Expressão prevista (n_cells_pred × n_genes).
    true_X:
        Expressão observada do target (n_cells_true × n_genes).
    n_pairs:
        Número de pares de genes amostrados.
    n_cells:
        Tamanho máximo da sub-amostra de células por lado.
    p:
        Expoente fracionário do variograma (padrão 0.5).
    seed:
        Semente do RNG para reprodutibilidade.

    Returns
    -------
    float
        MSE do variograma. 0 = estrutura co-expressional idêntica; maior = mais distante.
    """
    A, B = _to_dense(pred_X), _to_dense(true_X)
    rng = np.random.default_rng(seed)

    A = A[rng.choice(A.shape[0], min(n_cells, A.shape[0]), replace=False)]
    B = B[rng.choice(B.shape[0], min(n_cells, B.shape[0]), replace=False)]

    G = A.shape[1]
    i = rng.integers(0, G, n_pairs)
    j = rng.integers(0, G, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]

    # Variograma fracionário para cada par: E[ |x_i - x_j|^p ] ao longo das células
    va = (np.abs(A[:, i] - A[:, j]) ** p).mean(axis=0)
    vb = (np.abs(B[:, i] - B[:, j]) ** p).mean(axis=0)

    return float(((va - vb) ** 2).mean())


# ===========================================================================
# Normalização para o leaderboard (função skill do código oficial)
# ===========================================================================

def skill(
    value: float,
    floor: float,
    ceiling: float,
    lower_is_better: bool = False,
) -> float:
    """Normaliza uma métrica bruta para (0, 1] via mapa hiperbólico.

    Mapeamento (idêntico ao oficial):
        d(value)  = |value - ceiling|  (cresce à medida que value piora)
        d(floor)  = |ceiling - floor|  (distância fixa de "1 unidade de piso")
        skill     = d(floor) / (d(floor) + d(value))    [clampado em 1.0]

    Interpretação
        1.0  → atinge ou supera o ceiling (half-split oracle)
        0.5  → igual ao floor (baseline copy_last)
        →0   → arbitrariamente pior que o floor (sem atingir 0 exatamente)

    Parameters
    ----------
    value:
        Valor bruto da métrica.
    floor:
        Valor do baseline de referência (ex.: copy_last).
    ceiling:
        Valor attainable do ceiling (ex.: split-half oracle).
    lower_is_better:
        True para métricas como MMD e variogram_score.

    Returns
    -------
    float
        Skill score normalizado. NaN se value, floor ou ceiling não forem finitos.

    Examples
    --------
    >>> skill(0.5, floor=0.0, ceiling=0.846)          # DES parcialmente correto
    0.7182...
    >>> skill(0.04, floor=0.0836, ceiling=0.00406, lower_is_better=True)
    0.6926...
    """
    if not all(np.isfinite([value, floor, ceiling])):
        return float("nan")

    d_floor = abs(ceiling - floor)
    if d_floor < 1e-12:
        return float("nan")

    d = (value - ceiling) if lower_is_better else (ceiling - value)
    denom = d_floor + d
    if denom <= 1e-9:
        return 1.0
    return float(min(d_floor / denom, 1.0))


# ===========================================================================
# Função unificada
# ===========================================================================

def evaluate_all(
    pred_X: Matrix,
    true_X: Matrix,
    ref_X: Matrix,
    # Parâmetros de de_score / de_genes
    alpha: float = 0.05,
    min_lfc: float = MIN_LFC,
    # Parâmetros de mmd_unbiased
    mmd_n: int = 2_000,
    mmd_n_pc: int = 30,
    mmd_seed: int = 0,
    mmd_scales: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0),
    # Parâmetros de variogram_score
    vs_n_pairs: int = 20_000,
    vs_n_cells: int = 1_500,
    vs_p: float = 0.5,
    vs_seed: int = 0,
) -> dict:
    """Executa as quatro métricas oficiais da Task 1 de uma só vez.

    Parameters
    ----------
    pred_X:
        Expressão prevista (n_cells_pred × n_genes). log1p-normalizada.
    true_X:
        Expressão observada do target (n_cells_true × n_genes). log1p-normalizada.
    ref_X:
        Expressão da condição de referência (n_cells_ref × n_genes). log1p-normalizada.
    alpha:
        Nível FDR para seleção de genes DE.
    min_lfc:
        Piso de efeito mínimo em log-expressão.
    mmd_n:
        Sub-amostra máxima de células para MMD.
    mmd_n_pc:
        Componentes PCA para o espaço de avaliação do MMD.
    mmd_seed:
        Semente para sub-amostragem do MMD.
    mmd_scales:
        Escalas de largura de banda para o MMD multi-kernel.
    vs_n_pairs:
        Número de pares de genes para o variogram score.
    vs_n_cells:
        Sub-amostra máxima de células para o variogram score.
    vs_p:
        Expoente fracionário do variograma.
    vs_seed:
        Semente para sub-amostragem do variogram score.

    Returns
    -------
    dict com as chaves:
        ``de_score``         — DES (float). NaN se o ground truth não tem genes DE.
        ``de_score_detail``  — dict completo retornado por de_score().
        ``de_direction``     — DCS (float).
        ``mmd_unbiased``     — MMD² (float). Menor é melhor.
        ``variogram_score``  — VS (float). Menor é melhor.

    Examples
    --------
    >>> import anndata, numpy as np
    >>> rng = np.random.default_rng(42)
    >>> ref   = rng.lognormal(0, 1, (300, 500)).astype(np.float32)
    >>> true  = ref + rng.normal(0, 0.2, ref.shape)
    >>> pred  = ref + rng.normal(0, 0.25, ref.shape)
    >>> scores = evaluate_all(pred, true, ref)
    >>> scores.keys()
    dict_keys(['de_score', 'de_score_detail', 'de_direction', 'mmd_unbiased', 'variogram_score'])
    """
    des_detail = de_score(pred_X, true_X, ref_X, alpha=alpha, min_lfc=min_lfc)
    dcs = de_direction(pred_X, true_X, ref_X)
    mmd = mmd_unbiased(pred_X, true_X, n=mmd_n, n_pc=mmd_n_pc, seed=mmd_seed, scales=mmd_scales)
    vs = variogram_score(pred_X, true_X, n_pairs=vs_n_pairs, n_cells=vs_n_cells, p=vs_p, seed=vs_seed)

    return {
        "de_score":        des_detail["score"],   # DES  — 25%, higher is better
        "de_score_detail": des_detail,            # info auxiliar: n_up, n_dn, raw, chance…
        "de_direction":    dcs,                   # DCS  — 25%, higher is better
        "mmd_unbiased":    mmd,                   # MMD  — 30%, lower is better
        "variogram_score": vs,                    # CSS  — 20%, lower is better
    }


# ===========================================================================
# Exemplo de uso
# ===========================================================================

if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("Virtual Embryo Challenge — Task 1 Metrics")
    print("Exemplo sintético com dados aleatórios log1p-like")
    print("=" * 60)

    rng = np.random.default_rng(0)
    n_ref, n_true, n_pred = 400, 350, 300
    n_genes = 2_000   # mantido pequeno para o exemplo rodar rápido

    # Simula dados log1p-normalizados (valores típicos: 0–8)
    ref_X  = rng.lognormal(mean=0.5, sigma=1.0, size=(n_ref, n_genes)).astype(np.float32)
    # Ground truth: referência + um shift real em ~5% dos genes
    delta  = np.zeros(n_genes)
    de_idx = rng.choice(n_genes, 100, replace=False)
    delta[de_idx] = rng.normal(0, 0.8, 100)
    true_X = np.clip(ref_X[:n_true] + delta, 0, None).astype(np.float32)

    # Cenário 1: previsão razoável (ref + delta com ruído)
    pred_good = np.clip(ref_X[:n_pred] + delta * 0.7 + rng.normal(0, 0.15, (n_pred, n_genes)), 0, None).astype(np.float32)

    # Cenário 2: previsão ingênua (apenas copia a referência)
    pred_copy = ref_X[:n_pred].copy()

    for label, pred in [("Previsão razoável", pred_good), ("Copy (baseline)", pred_copy)]:
        print(f"\n--- {label} ---")
        scores = evaluate_all(pred, true_X, ref_X)
        des = scores["de_score"]
        det = scores["de_score_detail"]
        print(f"  DES  (de_score)      = {des:+.4f}  [25%]  raw={det['raw']:.4f}  chance={det['chance']:.4f}  n_DE={det['n_true']}")
        print(f"  DCS  (de_direction)  = {scores['de_direction']:+.4f}  [25%]")
        print(f"  MMD  (mmd_unbiased)  = {scores['mmd_unbiased']:.6f}  [30%]  (lower is better)")
        print(f"  CSS  (variogram)     = {scores['variogram_score']:.6f}  [20%]  (lower is better)")

    # Demonstração do skill()
    print("\n--- Skill score (normalização para o leaderboard) ---")
    print("  Âncoras Task 1 / E10.5 — verifique em virtualembryo.ai/challenge/evaluation")
    anchors = {
        "de_score":        {"floor": 0.00,    "ceiling": 0.846,    "lower": False},
        "de_direction":    {"floor": 0.00,    "ceiling": 0.790,    "lower": False},
        "mmd_unbiased":    {"floor": 0.0836,  "ceiling": 0.00406,  "lower": True},
        "variogram_score": {"floor": 0.00522, "ceiling": 0.000158, "lower": True},
    }
    weights = {
        "de_score": 0.25, "de_direction": 0.25,
        "mmd_unbiased": 0.30, "variogram_score": 0.20,
    }
    scores_good = evaluate_all(pred_good, true_X, ref_X)
    weighted_skill = 0.0
    for key, anc in anchors.items():
        v = scores_good[key]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            sk = float("nan")
        else:
            sk = skill(v, floor=anc["floor"], ceiling=anc["ceiling"], lower_is_better=anc["lower"])
        w = weights[key]
        weighted_skill += w * (sk if np.isfinite(sk) else 0.0)
        print(f"  {key:20s}: valor={v:.5f}  skill={sk:.4f}  (peso {int(w*100)}%)")
    print(f"\n  Score ponderado estimado: {weighted_skill:.4f}")