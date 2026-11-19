"""Unsupervised anomaly detectors with a common ``fit`` / ``score`` interface.

Every detector is fitted on training features and returns scores where
**higher = more anomalous**. Scaling is part of each detector and is fitted on
training data only.

Implemented (and tested):

* :class:`IsolationForestDetector`  - ``-decision_function`` of an Isolation Forest
* :class:`KMeansDistanceDetector`   - distance to the nearest K-Means centroid in PCA space
* :class:`MLPAutoencoderDetector`   - reconstruction error of a scikit-learn MLP autoencoder
* :class:`LOFDetector`, :class:`OneClassSVMDetector`, :class:`GMMDetector`

Optional (not executed in this repository's tests - TensorFlow not installed):

* :class:`KerasAutoencoderDetector` - dense Keras autoencoder (the architecture of
  the original TPU script), importable only when TensorFlow is available.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


def _as_array(X) -> np.ndarray:
    return X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)


class BaseDetector:
    """Standardises features (fitted on training data) before the model."""

    name = "base"

    def fit(self, X) -> "BaseDetector":
        Xa = _as_array(X)
        self.scaler_ = StandardScaler().fit(Xa)
        self._fit(self.scaler_.transform(Xa))
        return self

    def score(self, X) -> np.ndarray:
        if not hasattr(self, "scaler_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted")
        return np.asarray(self._score(self.scaler_.transform(_as_array(X))), dtype=float)

    def _fit(self, Xs: np.ndarray) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    def _score(self, Xs: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def _fit(self, Xs):
        self.model_ = IsolationForest(n_estimators=self.n_estimators, random_state=self.random_state).fit(Xs)

    def _score(self, Xs):
        return -self.model_.decision_function(Xs)


class KMeansDistanceDetector(BaseDetector):
    """PCA (95 % variance) -> K-Means; score = Euclidean distance to the nearest centroid."""

    name = "kmeans_distance"

    def __init__(self, n_clusters: int = 8, pca_variance: float = 0.95, random_state: int = 42):
        self.n_clusters = n_clusters
        self.pca_variance = pca_variance
        self.random_state = random_state

    def _fit(self, Xs):
        self.pca_ = PCA(n_components=self.pca_variance, random_state=self.random_state).fit(Xs)
        self.model_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.model_.fit(self.pca_.transform(Xs))

    def _score(self, Xs):
        return self.model_.transform(self.pca_.transform(Xs)).min(axis=1)


class MLPAutoencoderDetector(BaseDetector):
    """MLPRegressor trained to reproduce its input; score = mean squared reconstruction error."""

    name = "mlp_autoencoder"

    def __init__(self, max_iter: int = 200, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state

    def _fit(self, Xs):
        d = Xs.shape[1]
        h1, h2 = max(2, d // 2), max(2, d // 4)
        self.model_ = MLPRegressor(hidden_layer_sizes=(h1, h2, h1), max_iter=self.max_iter,
                                   random_state=self.random_state, early_stopping=True,
                                   n_iter_no_change=10)
        self.model_.fit(Xs, Xs)

    def _score(self, Xs):
        return ((Xs - self.model_.predict(Xs)) ** 2).mean(axis=1)


class LOFDetector(BaseDetector):
    """Local Outlier Factor in novelty mode (fit on train, score new visits)."""

    name = "lof"

    def __init__(self, n_neighbors: int = 20):
        self.n_neighbors = n_neighbors

    def _fit(self, Xs):
        k = max(2, min(self.n_neighbors, Xs.shape[0] - 1))
        self.model_ = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xs)

    def _score(self, Xs):
        return -self.model_.score_samples(Xs)


class OneClassSVMDetector(BaseDetector):
    name = "one_class_svm"

    def __init__(self, nu: float = 0.14, gamma: str | float = "auto"):
        self.nu = nu
        self.gamma = gamma

    def _fit(self, Xs):
        self.model_ = OneClassSVM(nu=self.nu, kernel="rbf", gamma=self.gamma).fit(Xs)

    def _score(self, Xs):
        return -self.model_.decision_function(Xs)


class GMMDetector(BaseDetector):
    """Negative log-likelihood under a Gaussian mixture."""

    name = "gmm"

    def __init__(self, n_components: int = 4, covariance_type: str = "diag", random_state: int = 42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state

    def _fit(self, Xs):
        self.model_ = GaussianMixture(n_components=self.n_components, covariance_type=self.covariance_type,
                                      random_state=self.random_state, reg_covar=1e-4).fit(Xs)

    def _score(self, Xs):
        return -self.model_.score_samples(Xs)


class KerasAutoencoderDetector(BaseDetector):
    """Dense autoencoder (input -> 64 -> 32 -> 64 -> input), as in the original TPU script.

    Requires TensorFlow, which is an optional dependency. NOT executed in this
    repository's test suite.
    """

    name = "keras_autoencoder"

    def __init__(self, intermediate_dim: int = 64, encoding_dim: int = 32, epochs: int = 50,
                 batch_size: int = 512, learning_rate: float = 1e-3, random_state: int = 42):
        self.intermediate_dim = intermediate_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    @staticmethod
    def _tf():
        try:
            import tensorflow as tf  # noqa: WPS433 - optional dependency
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "KerasAutoencoderDetector needs TensorFlow (pip install tensorflow); "
                "use MLPAutoencoderDetector for a dependency-free alternative"
            ) from exc
        return tf

    def _fit(self, Xs):  # pragma: no cover - needs tensorflow
        tf = self._tf()
        tf.keras.utils.set_random_seed(self.random_state)
        layers = tf.keras.layers
        inp = tf.keras.Input(shape=(Xs.shape[1],))
        x = layers.Dense(self.intermediate_dim, activation="relu")(inp)
        x = layers.Dense(self.encoding_dim, activation="relu")(x)
        x = layers.Dense(self.intermediate_dim, activation="relu")(x)
        out = layers.Dense(Xs.shape[1], activation="linear")(x)
        self.model_ = tf.keras.Model(inp, out)
        self.model_.compile(optimizer=tf.keras.optimizers.Adam(self.learning_rate), loss="mse")
        self.model_.fit(Xs, Xs, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, verbose=0)

    def _score(self, Xs):  # pragma: no cover - needs tensorflow
        return ((Xs - self.model_.predict(Xs, verbose=0)) ** 2).mean(axis=1)


DETECTORS = {
    cls.name: cls
    for cls in (IsolationForestDetector, KMeansDistanceDetector, MLPAutoencoderDetector,
                LOFDetector, OneClassSVMDetector, GMMDetector, KerasAutoencoderDetector)
}


def make_detector(name: str, **kwargs) -> BaseDetector:
    try:
        return DETECTORS[name](**kwargs)
    except KeyError as exc:
        raise ValueError(f"unknown detector {name!r}; choose from {sorted(DETECTORS)}") from exc
