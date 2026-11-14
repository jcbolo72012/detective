"""A small LSTM written in NumPy: forward pass, backpropagation through time, Adam. No PyTorch.

Sized for this app: one layer, a few dozen hidden units, sequences of a few years of monthly values. It trains
in seconds on a CPU and needs nothing beyond NumPy, so it runs on Streamlit Community Cloud as is.

    net = LSTM(n_in=1, n_hidden=16, seed=0)
    net.fit(X, Y, epochs=200)          # X: (batch, time, n_in), Y: (batch, 1)
    yhat = net.predict(X)              # (batch, 1)

Self-test (gradient check against finite differences):  python lstm_numpy.py
"""
import numpy as np


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))  # stable for large |x|


class LSTM:
    """One LSTM layer reading a sequence, then a linear head on the last hidden state.

    Gates are stacked in the order input, forget, cell candidate, output: W is (n_in + n_hidden, 4H)."""

    def __init__(self, n_in=1, n_hidden=16, seed=0, dtype=np.float64):
        rng = np.random.default_rng(seed)
        H, D = n_hidden, n_in
        scale = 1.0 / np.sqrt(D + H)
        self.H, self.D, self.dtype = H, D, dtype
        self.p = {
            "W": rng.uniform(-scale, scale, (D + H, 4 * H)).astype(dtype),
            "b": np.zeros(4 * H, dtype),
            "Wy": rng.uniform(-1 / np.sqrt(H), 1 / np.sqrt(H), (H, 1)).astype(dtype),
            "by": np.zeros(1, dtype),
        }
        self.p["b"][H:2 * H] = 1.0  # forget-gate bias 1: remember by default (standard trick)
        self._adam = None

    # ------------------------------------------------------------------ forward / backward
    def forward(self, X, keep=False):
        """X (B, T, D) -> yhat (B, 1). keep=True stores what backward() needs."""
        X = np.asarray(X, self.dtype)
        B, T, _ = X.shape
        H, W, b = self.H, self.p["W"], self.p["b"]
        h = np.zeros((B, H), self.dtype)
        c = np.zeros((B, H), self.dtype)
        cache = []
        for t in range(T):
            z = np.concatenate([X[:, t], h], axis=1)
            a = z @ W + b
            i, f = _sigmoid(a[:, :H]), _sigmoid(a[:, H:2 * H])
            g, o = np.tanh(a[:, 2 * H:3 * H]), _sigmoid(a[:, 3 * H:])
            c_prev = c
            c = f * c_prev + i * g
            tc = np.tanh(c)
            h = o * tc
            if keep:
                cache.append((z, i, f, g, o, c_prev, tc))
        y = h @ self.p["Wy"] + self.p["by"]
        if keep:
            self._cache = (cache, h)
        return y

    def backward(self, dy):
        """dy = dLoss/dyhat (B, 1), after forward(keep=True). Returns gradients with the shapes of self.p."""
        cache, h_last = self._cache
        H, W = self.H, self.p["W"]
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        grads["Wy"] = h_last.T @ dy
        grads["by"] = dy.sum(axis=0)
        dh = dy @ self.p["Wy"].T
        dc = np.zeros_like(dh)
        for z, i, f, g, o, c_prev, tc in reversed(cache):
            do = dh * tc
            dc = dc + dh * o * (1 - tc ** 2)
            di, df, dg = dc * g, dc * c_prev, dc * i
            da = np.concatenate([di * i * (1 - i), df * f * (1 - f), dg * (1 - g ** 2), do * o * (1 - o)], axis=1)
            grads["W"] += z.T @ da
            grads["b"] += da.sum(axis=0)
            dz = da @ W.T
            dh = dz[:, self.D:]
            dc = dc * f
        return grads

    # ------------------------------------------------------------------ training
    def loss_and_grads(self, X, Y):
        yhat = self.forward(X, keep=True)
        err = yhat - np.asarray(Y, self.dtype)
        n = err.shape[0]
        return float(np.mean(err ** 2)), self.backward(2 * err / n)

    def fit(self, X, Y, epochs=200, lr=0.01, batch=64, clip=5.0, seed=0, patience=None, X_val=None, Y_val=None):
        """Adam on mean squared error, shuffled minibatches, gradient-norm clipping.
        With X_val/Y_val and patience, stops when validation loss hasn't improved for `patience` epochs and
        restores the best weights. Returns the list of training losses per epoch."""
        X, Y = np.asarray(X, self.dtype), np.asarray(Y, self.dtype)
        rng = np.random.default_rng(seed)
        if self._adam is None:
            self._adam = {"t": 0, "m": {k: np.zeros_like(v) for k, v in self.p.items()},
                          "v": {k: np.zeros_like(v) for k, v in self.p.items()}}
        st, b1, b2, eps = self._adam, 0.9, 0.999, 1e-8
        history, best, best_p, bad = [], np.inf, None, 0
        n = len(X)
        for _ in range(epochs):
            order = rng.permutation(n)
            total = 0.0
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                loss, g = self.loss_and_grads(X[idx], Y[idx])
                total += loss * len(idx)
                norm = np.sqrt(sum(float((v ** 2).sum()) for v in g.values()))
                if norm > clip:
                    g = {k: v * (clip / norm) for k, v in g.items()}
                st["t"] += 1
                for k in self.p:
                    st["m"][k] = b1 * st["m"][k] + (1 - b1) * g[k]
                    st["v"][k] = b2 * st["v"][k] + (1 - b2) * g[k] ** 2
                    mh = st["m"][k] / (1 - b1 ** st["t"])
                    vh = st["v"][k] / (1 - b2 ** st["t"])
                    self.p[k] -= lr * mh / (np.sqrt(vh) + eps)
            history.append(total / n)
            if patience and X_val is not None and len(X_val):
                val = float(np.mean((self.predict(X_val) - np.asarray(Y_val, self.dtype)) ** 2))
                if val < best - 1e-9:
                    best, best_p, bad = val, {k: v.copy() for k, v in self.p.items()}, 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_p is not None:
            self.p = best_p
        return history

    def predict(self, X):
        return self.forward(X)

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))


def gradient_check(n_in=2, n_hidden=5, T=7, B=4, seed=1, eps=1e-5):
    """Largest relative error, per parameter tensor, between backprop and centred finite differences
    (float64): ||numeric - analytic|| / ||numeric + analytic||. Around 1e-8 or smaller means backprop is right."""
    rng = np.random.default_rng(seed)
    net = LSTM(n_in, n_hidden, seed=seed)
    X, Y = rng.normal(size=(B, T, n_in)), rng.normal(size=(B, 1))
    _, g = net.loss_and_grads(X, Y)
    worst = 0.0
    for k, v in net.p.items():
        flat, num = v.reshape(-1), np.zeros(v.size)
        for j in range(flat.size):
            old = flat[j]
            flat[j] = old + eps
            lp = float(np.mean((net.forward(X) - Y) ** 2))
            flat[j] = old - eps
            lm = float(np.mean((net.forward(X) - Y) ** 2))
            flat[j] = old
            num[j] = (lp - lm) / (2 * eps)
        ana = g[k].reshape(-1)
        worst = max(worst, float(np.linalg.norm(num - ana) / max(1e-12, np.linalg.norm(num + ana))))
    return worst


if __name__ == "__main__":
    err = gradient_check()
    print(f"gradient check: max relative error {err:.2e} ({'OK' if err < 1e-6 else 'FAILED'})")
