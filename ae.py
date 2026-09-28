"""
Autoencoder con regla local XOR (Peña, Lloret, Marco 2026) y su traducción a Ł3+T.

Variantes
  paper : réplica (neuronas {0,1}, pesos reales, eta_dec=0.05, eta_enc=0.02).
  l3bin : misma arquitectura en el modelo lógico de dos capas:
            ℓ-grupo ℤ  -> integración (pesos y sesgos enteros, sumas exactas)
            Ł3+T       -> todas las señales: error, producto Hebb, signo del encoder
          Escala x200 respecto al paper (eta_dec=10, eta_enc=4): la activación escalón
          es invariante de escala, así que la dinámica es la misma salvo el redondeo inicial.
  l3ter : como l3bin pero la capa oculta es ternaria h = q(u, θ) ∈ {-1,0,1}.

Señales en Ł3 (ver l3.py):
  e   = (x → f) → (T ⊙ (f → x))        error con signo
  h·e = (h ⊕ e) → (h ∧ e)              término Hebb (signed-XNOR)
  s   = q(e B)                         dirección del error en la capa oculta
"""
import gzip, time, json, sys
import numpy as np
import l3

def load(kind):
    p = "data/" + ("train" if kind == "train" else "t10k")
    X = np.frombuffer(gzip.open(p + "-images-idx3-ubyte.gz").read(), np.uint8, offset=16)
    y = np.frombuffer(gzip.open(p + "-labels-idx1-ubyte.gz").read(), np.uint8, offset=8)
    return (X.reshape(-1, 784) > 127).astype(np.float64), y.astype(np.int64)

def metrics(x, f):
    tp = np.sum((x == 1) & (f == 1)); tn = np.sum((x == 0) & (f == 0))
    fp = np.sum((x == 0) & (f == 1)); fn = np.sum((x == 1) & (f == 0))
    acc = (tp + tn) / x.size
    bal = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    f1 = 2 * tp / (2 * tp + fp + fn)
    return dict(bit_acc=100 * acc, bal_acc=100 * bal, f1=f1)

class AE:
    def __init__(self, variant, n_hid=1024, seed=0, theta_frac=0.5):
        self.v = variant
        rng = np.random.default_rng(seed)
        self.theta_frac = theta_frac
        if variant == "paper":
            self.W = 0.01 * rng.standard_normal((784, n_hid))
            self.V = 0.01 * rng.standard_normal((n_hid, 784))
            self.B = rng.standard_normal((784, n_hid))
            self.bh = np.zeros(n_hid); self.bo = np.zeros(784)
            self.eta = dict(dec=0.05, enc=0.02, bo=0.005, bh=0.005)
        else:  # enteros (float64 con valores enteros: BLAS exacto hasta 2^53)
            self.W = np.round(2 * rng.standard_normal((784, n_hid)))
            self.V = np.round(2 * rng.standard_normal((n_hid, 784)))
            self.B = rng.choice([-1.0, 1.0], size=(784, n_hid))
            self.bh = np.zeros(n_hid); self.bo = np.zeros(784)
            self.eta = dict(dec=10, enc=4, bo=1, bh=1)
        self.sign_ok = []   # fracción de bits con error cuyo potencial se mueve en la dirección de e

    def hidden(self, x):
        u = x @ self.W + self.bh
        if self.v == "l3ter":
            # zona muerta relativa: θ = fracción de la mediana de |u| por patrón (control de ganancia)
            th = self.theta_frac * np.median(np.abs(u), axis=1, keepdims=True)
            return l3.q(u, th).astype(np.float64)
        return (u >= 0).astype(np.float64)

    def out(self, h):
        return (h @ self.V + self.bo >= 0).astype(np.float64)

    def step(self, x, check=False):
        h = self.hidden(x); f = self.out(h)
        if self.v == "paper":
            e = x - f                                   # XOR + bit objetivo = error con signo
            s = np.sign(e @ self.B)
        else:
            e = l3.signed_error(x, f).astype(np.float64)                   # fórmula Ł3+T
            s = l3.q(e @ self.B).astype(np.float64)                        # integración en ℤ + cuantización
            if check:                                   # el término Hebb es el producto Ł3
                j = np.random.default_rng(1).integers(0, h.shape[1], 64)
                assert np.array_equal(l3.signed_product(h[:, j][:, :, None], e[:, None, :]).sum(0),
                                      (h[:, j].T @ e))
        dV = self.eta["dec"] * (h.T @ e)
        if check:  # lema: Δu_i^n = eta * (|h^n|·e_i^n + Σ_{m≠n} <h^n,h^m> e_i^m) + Δb
            du = h @ dV + self.eta["bo"] * e.sum(0)
            m = e != 0
            self.sign_ok.append(float(np.mean(np.sign(du[m]) == e[m])))
        self.V += dV
        self.W += self.eta["enc"] * (x.T @ s)
        self.bo += self.eta["bo"] * e.sum(0)
        self.bh += self.eta["bh"] * s.sum(0)
        return e

    def reconstruct(self, x, bs=2000):
        return np.vstack([self.out(self.hidden(x[i:i + bs])) for i in range(0, len(x), bs)])

    def codes(self, x, bs=2000):
        return np.vstack([self.hidden(x[i:i + bs]) for i in range(0, len(x), bs)])

def probe(Htr, ytr, Hte, yte, epochs=5, eta=0.05, bs=256, seed=0):
    """Perceptrón multiclase local (paper §2.4)."""
    rng = np.random.default_rng(seed)
    Wc = np.zeros((Htr.shape[1] + 1, 10))
    A = np.hstack([Htr, np.ones((len(Htr), 1))]); At = np.hstack([Hte, np.ones((len(Hte), 1))])
    for _ in range(epochs):
        for i in np.array_split(rng.permutation(len(A)), len(A) // bs):
            p = np.argmax(A[i] @ Wc, 1); wrong = p != ytr[i]
            D = np.zeros((len(i), 10))
            D[np.arange(len(i)), ytr[i]] += wrong; D[np.arange(len(i)), p] -= wrong
            Wc += eta * A[i].T @ D
    return 100 * np.mean(np.argmax(At @ Wc, 1) == yte)

def run(variant, epochs=5, bs=256, seed=0, theta_frac=0.5):
    Xtr, ytr = load("train"); Xte, yte = load("test")
    ae = AE(variant, seed=seed, theta_frac=theta_frac)
    before = metrics(Xte, ae.reconstruct(Xte))
    rng = np.random.default_rng(seed); t0 = time.time()
    hist = []
    for ep in range(epochs):
        for k, i in enumerate(np.array_split(rng.permutation(len(Xtr)), len(Xtr) // bs)):
            ae.step(Xtr[i], check=(k % 20 == 0))
        m = metrics(Xte, ae.reconstruct(Xte)); hist.append(m)
        print(f"[{variant}] epoch {ep+1}: {m}  ({time.time()-t0:.0f}s)", flush=True)
    Hte = ae.codes(Xte)
    res = dict(variant=variant, before=before, after=hist[-1], history=hist,
               probe_acc=probe(ae.codes(Xtr), ytr, Hte, yte),
               sign_ok_mean=float(np.mean(ae.sign_ok)),
               hidden_frac={str(v): float(np.mean(Hte == v)) for v in (-1, 0, 1)})
    print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1), flush=True)
    return res

if __name__ == "__main__":
    l3.verify()
    out = [run(v) for v in sys.argv[1:]]
    json.dump(out, open("results_" + "_".join(sys.argv[1:]) + ".json", "w"), indent=1)
