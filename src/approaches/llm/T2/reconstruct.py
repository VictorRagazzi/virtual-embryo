"""
Decodificação: embedding extrapolado → expressão gênica.

Problema: não existe E10.5 real para treinar a reconstrução.
Solução: treinar um decoder E8.5 + E9.5 e aplicar nos embeddings extrapolados.
Ridge Regression é preferido — regularização evita overfitting em n_genes >> n_cells.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


class EmbeddingToExpressionDecoder:
    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = MultiOutputRegressor(
            Ridge(alpha=alpha),
            n_jobs=-1,
        )

    def fit(self, embeddings: np.ndarray, expression: np.ndarray):
        """
        embeddings: (n_cells, embed_dim)
        expression: (n_cells, n_hvgs) — expressão log-normalizada
        """
        X = self.scaler_x.fit_transform(embeddings)
        y = self.scaler_y.fit_transform(expression)
        self.model.fit(X, y)
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        X = self.scaler_x.transform(embeddings)
        y_pred = self.model.predict(X)
        expr_pred = self.scaler_y.inverse_transform(y_pred)
        return np.clip(expr_pred, 0, None)  # expressão ≥ 0