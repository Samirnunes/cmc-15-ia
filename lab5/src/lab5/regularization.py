import numpy as np
from scipy.spatial.distance import cdist
from sklearn.exceptions import NotFittedError
from abc import ABC

class RegularizationLayer:
    def __init__(self, random_state: int = 0):
        self._rand = np.random.RandomState(random_state)
        self._n_centers: int = None
        self._centers: np.ndarray = None
        self._sigma: float = None
        self._fitted = False
    
    def fit(self, X: np.ndarray):
        self._n_centers = X.shape[0]
        self._centers = self._calculate_centers(X)
        self._sigma = self._calculate_sigma(X)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray):
        if self._fitted:
            return RegularizationLayer._gaussian_rbf(X, self._centers, self._sigma)
        raise NotFittedError
    
    def _calculate_centers(self, X: np.ndarray):
        return X
    
    def _calculate_sigma(self, X: np.ndarray):
        return max(-cdist(X, self._centers, "sqeuclidean")) / np.sqrt(2 * self._n_centers)

    @staticmethod
    def _gaussian_rbf(X: np.ndarray, center: np.ndarray, sigma: float):
        return np.exp(-cdist(X, center, "sqeuclidean") / (2 * sigma**2))
    
class BaseRegularizationNetwork(ABC):
    def __init__(self, random_state: int = 0):
        self.random_state: int = random_state
        self.layer: RegularizationLayer = None
        self.weights: np.ndarray = None
        self.fitted: bool = False
        
    def predict(self, X):
        if self.fitted:
            phi = self.layer.predict(X)
            return phi @ self.weights
        raise NotFittedError
    
class RegularizationNetwork(BaseRegularizationNetwork):
    def __init__(self, random_state: int = 0):
        super().__init__(random_state)

    def fit(self, X_train, y_train):
        self.layer = RegularizationLayer(self.random_state).fit(X_train)
        phi: np.ndarray = self.layer.predict(X_train)
        self.weights: np.ndarray = np.linalg.inv(phi) @ y_train
        self.fitted = True
        return self
    
class GreenRegularizationNetwork(BaseRegularizationNetwork):
    def __init__(self, n_neurons: int, lambda_param: int, random_state: int = 0):
        super().__init__(n_neurons, random_state)
        self.lambda_param = lambda_param

    def fit(self, X_train, y_train):
        self.layer = RegularizationLayer(self.random_state).fit(X_train)
        G = self.layer.predict(X_train)
        self.weights = self._calculate_green_inverse(G) @ y_train
        self.fitted = True
        return self

    def _calculate_green_inverse(self, G: np.ndarray):
        return np.linalg.inv(G + self.lambda_param * np.identity(G.shape[0]))
