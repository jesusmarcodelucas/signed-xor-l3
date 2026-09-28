"""
Ł3 + T (Słupecki) sobre {-1, 0, 1} (codificación desplazada), vectorizado.
  ¬x = -x          x ⊕ y = min(1, x+y+1)     x ⊙ y = max(-1, x+y-1)
  x → y = min(1, 1-x+y)     ∧ = min,  ∨ = max,  T = 0
Fórmulas mínimas (búsqueda exhaustiva BFS, base {x,y,T,¬,⊕,⊙,∧,∨,→}):
  producto con signo   x·y         = (x ⊕ y) → (x ∧ y)          [sin T]
  error con signo      clip(x - f) = (x → f) → (T ⊙ (f → x))    [requiere T]
"""
import itertools
import numpy as np

V = (-1, 0, 1)

def neg(x):      return -x
def oplus(x, y): return np.minimum(1, x + y + 1)
def odot(x, y):  return np.maximum(-1, x + y - 1)
def imp(x, y):   return np.minimum(1, 1 - x + y)
def wedge(x, y): return np.minimum(x, y)
def vee(x, y):   return np.maximum(x, y)
def iff(x, y):   return wedge(imp(x, y), imp(y, x))
def T(x):        return np.zeros_like(x)

def signed_product(x, y):
    """Signed-XNOR / producto con 0 absorbente: (x ⊕ y) → (x ∧ y)."""
    return imp(oplus(x, y), wedge(x, y))

def signed_error(x, f):
    """Error con signo: +1 falta disparo, -1 disparo espurio, 0 acuerdo.
    (x → f) → (T ⊙ (f → x))"""
    return imp(imp(x, f), odot(T(x), imp(f, x)))

def q(u, theta=0):
    """Cuantización ℓ-grupo ℤ -> {-1,0,1} con zona muerta simétrica (theta=0 -> signo)."""
    return (u > theta).astype(np.int64) - (u < -theta).astype(np.int64)

def truth_table(fun, arity=2):
    pts = np.array(list(itertools.product(V, repeat=arity))).T
    return tuple(int(v) for v in fun(*pts))

def search(targets, max_size=11, use_T=True):
    """Fórmula más corta para cada tabla objetivo (tuplas de 9 valores)."""
    P = list(itertools.product(V, repeat=2))
    X = tuple(p[0] for p in P); Y = tuple(p[1] for p in P)
    level = {X: "x", Y: "y"}
    if use_T: level[(0,) * 9] = "T"
    by_size, seen, found = {1: level}, dict(level), {}
    bins = [("⊕", oplus), ("⊙", odot), ("∧", wedge), ("∨", vee), ("→", imp)]
    def check(tt, s):
        for n, t in targets.items():
            if n not in found and t == tt: found[n] = s
    for tt, s in level.items(): check(tt, s)
    for size in range(2, max_size + 1):
        cur = {}
        for tt, s in by_size.get(size - 1, {}).items():
            nt = tuple(-v for v in tt)
            if nt not in seen: cur[nt] = f"¬{s}"
        for ls in range(1, size - 1):
            for ta, sa in by_size.get(ls, {}).items():
                a = np.array(ta)
                for tb, sb in by_size.get(size - 1 - ls, {}).items():
                    b = np.array(tb)
                    for sym, op in bins:
                        nt = tuple(int(v) for v in op(a, b))
                        if nt not in seen and nt not in cur: cur[nt] = f"({sa} {sym} {sb})"
        for tt, s in cur.items(): seen[tt] = s; check(tt, s)
        by_size[size] = cur
        if len(found) == len(targets): break
    return found

def verify():
    x, f = np.array(list(itertools.product(V, repeat=2))).T
    assert np.array_equal(signed_product(x, f), x * f)
    assert np.array_equal(signed_error(x, f), np.clip(x - f, -1, 1))
    # Lema del cuadrado: h·(h·e) = |h|·e  (base de la corrección de signo de la regla Hebb)
    h, e = x, f
    assert np.array_equal(signed_product(h, signed_product(h, e)), np.abs(h) * e)
    # Homeostasis: error nulo  <=>  equivalencia Ł3 plena
    assert np.array_equal(signed_error(x, f) == 0, iff(x, f) == 1)
    # Observación "dónde se necesita T": lectura de dos canales excluyentes (sin T)
    r = lambda a, b: odot(a, imp(b, a))
    assert [int(r(np.array(a), np.array(b))) for a, b in [(1, 0), (0, 1), (0, 0)]] == [1, -1, 0]
    # canal e+ con "apagado" = -1 (codificación extrema, estilo booleano): ¬(x → f), sin T
    xs, fs = np.array([-1, -1, 1, 1]), np.array([-1, 1, -1, 1])
    assert list(neg(imp(xs, fs))) == [-1, -1, 1, -1]
    return True

if __name__ == "__main__":
    print("identidades verificadas:", verify())
